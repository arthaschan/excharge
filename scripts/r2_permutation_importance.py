#!/usr/bin/env python3
"""R2 补跑: Token-Attn 置换特征重要性 (仅深度模型, 不含 TabPFN)。

背景:
  §4.3 的 Top5 置换重要性 (t2_max ΔPR-AUC=0.488 等) 最初在 r3_feature_importance.py 里
  与 TabPFN 一并计算, 但当时只保存了 3 次置换的均值 (np.mean), 没有保存方差/置信区间。
  为回应论文审计, 本脚本用**与原始完全一致的口径**重跑 Token-Attn 分支:
    - 模型: c1c2_tokenattn_model.pt (seed=42 单模型)
    - 每个特征随机打乱测试集该列 n_repeat=3 次, 取 ΔPR-AUC 均值
    - 随机种子: np.random.default_rng(0)  (与原始一致, 确定性可复现)
    - base PR-AUC = 0.9158 (对应 §4.3 消融表"全量 0.916")
  额外记录: 每个特征 3 次结果的 std / sem / 95% CI (t 分布, df=2)。

输出: docs/r2_permutation_importance.json
"""
import os, sys, pickle, warnings, json, time
warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '4'
import numpy as np
import torch
torch.set_num_threads(4)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_c1c2 as T

DATA = T.DATA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
D = T.D
FEAT_COLS = D['feat_cols']
Fte = D['test']['X_feat'].astype(np.float64)
yte = np.asarray(D['test']['y'])
Xseq_te, lte = T.pad(D['test']['X_tensor'])
Xseq_te_t = torch.FloatTensor(Xseq_te).to(device)
lte_t = torch.LongTensor(lte).to(device)

from sklearn.metrics import average_precision_score
from scipy import stats as scipy_stats  # t 分布临界值

# ---------- Token-Attn (seed42) ----------
ckpt = torch.load(f'{T.OUT}/c1c2_tokenattn_model.pt', map_location='cpu')
model = T.TokenAttnFusion(62).to(device)
model.load_state_dict(ckpt['state'])
model.eval()
print(f'loaded c1c2_tokenattn_model.pt (meta seed={ckpt["meta"]["seed"]}, '
      f'best PR-AUC={ckpt["meta"].get("PR-AUC")})', flush=True)

def ta_predict(F):
    with torch.no_grad():
        Ft = torch.FloatTensor(F.astype(np.float32)).to(device)
        return torch.softmax(model(Xseq_te_t, lte_t, Ft), 1)[:, 1].cpu().numpy()

def perm_importance(predict_fn, Fte, n_repeat=3, seed=0):
    """逐特征打乱该列, 返回 base + 每特征 [均值, std, sem, ci95, 单次列表]。
    与 r3_feature_importance.py 的打乱顺序完全一致: default_rng(0), 外层特征/内层重复。"""
    base = average_precision_score(yte, predict_fn(Fte))
    rng = np.random.default_rng(seed)
    n_feat = Fte.shape[1]
    mean = np.zeros(n_feat); std = np.zeros(n_feat); sem = np.zeros(n_feat)
    ci95 = np.zeros((n_feat, 2)); repeats = {}
    tcrit = float(scipy_stats.t.ppf(0.975, df=n_repeat - 1))  # df=2 → 4.3027
    for j in range(n_feat):
        ds = []
        for _ in range(n_repeat):
            Fp = Fte.copy()
            rng.shuffle(Fp[:, j])
            ds.append(base - average_precision_score(yte, predict_fn(Fp)))
        ds = np.asarray(ds)
        mean[j] = ds.mean()
        std[j] = ds.std(ddof=1)  # 样本标准差 (n=3)
        sem[j] = std[j] / np.sqrt(n_repeat)
        ci95[j, 0] = mean[j] - tcrit * sem[j]
        ci95[j, 1] = mean[j] + tcrit * sem[j]
        repeats[FEAT_COLS[j]] = [float(x) for x in ds]
    return base, mean, std, sem, ci95, repeats

print('Token-Attn permutation importance (62 feat x 3), recording mean/std/CI ...', flush=True)
t0 = time.time()
base_ta, mean_ta, std_ta, sem_ta, ci95_ta, repeats_ta = perm_importance(ta_predict, Fte)
print(f'  base PR-AUC={base_ta:.4f}, done {time.time()-t0:.0f}s', flush=True)

# ---------- 排序取 top ----------
order = np.argsort(mean_ta)[::-1]
top = []
for i in order[:10]:
    top.append({'feature': FEAT_COLS[i],
                'mean_dPR_AUC': round(float(mean_ta[i]), 4),
                'std': round(float(std_ta[i]), 4),
                'ci95': [round(float(ci95_ta[i, 0]), 4), round(float(ci95_ta[i, 1]), 4)]})

res = {
    'model': 'Token-Attn (seed=42, c1c2_tokenattn_model.pt)',
    'n_repeat': 3,
    'rng_seed': 0,
    'base_PR_AUC': float(base_ta),
    'note': 'ΔPR-AUC = base_PR_AUC - 打乱该特征后的 PR-AUC; 正值=重要; std/sem 为 3 次置换的样本标准差/标准误; ci95 为 t(0.975, df=2) 置信区间',
    't_critical_df2': 4.3027,
    'top10': top,
    'per_feature': {
        c: {
            'mean': float(mean_ta[i]),
            'std': float(std_ta[i]),
            'sem': float(sem_ta[i]),
            'ci95': [float(ci95_ta[i, 0]), float(ci95_ta[i, 1])],
            'repeats': repeats_ta[c],
        }
        for i, c in enumerate(FEAT_COLS)
    },
}
out_path = f'{T.OUT}/r2_permutation_importance.json'
json.dump(res, open(out_path, 'w'), indent=2, ensure_ascii=False)

# ---------- 控制台摘要 ----------
print('\n=== Token-Attn 置换重要性 Top5 (n=3 置换取平均, seed=0) ===', flush=True)
print(f'base PR-AUC = {base_ta:.4f}', flush=True)
print(f'{"特征":<14s}{"ΔPR-AUC(均值)":>14s}{"±std":>10s}{"95%CI":>22s}', flush=True)
for r in top[:5]:
    print(f'{r["feature"]:<14s}{r["mean_dPR_AUC"]:>14.4f}{r["std"]:>10.4f}'
          f'  [{r["ci95"][0]:.4f}, {r["ci95"][1]:.4f}]', flush=True)
print(f'\nSaved {out_path}', flush=True)
print('DONE', flush=True)
