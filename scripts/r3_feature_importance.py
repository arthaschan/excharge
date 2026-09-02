#!/usr/bin/env python3
"""R3: 双模型特征归因对齐 —— 置换特征重要性 (Token-Attn vs TabPFN)。

R3 叙事核心之一: 可解释深度模型(Token-Attn)与表格基础模型天花板(TabPFN)在"哪些特征重要"上
结论一致, 用同一套方法(置换重要性, PR-AUC 下降)对两个模型做特征归因, 再与注意力图(fig3)对照。

做法: 固定模型, 逐特征(62)打乱测试集该列(3 次取平均), 算 PR-AUC 相对基线的下降。
输出:
  docs/r3_feature_importance.json
  docs/r3_figs/feature_importance_alignment.png (散点: tokenattn vs tabpfn 重要性 + 相关系数)
"""
import os, sys, pickle, warnings, json, time
warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '4'
import numpy as np
import torch
torch.set_num_threads(4)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_c1c2 as T

_ROOT = T.OUT  # docs/
DATA = T.DATA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
D = T.D
FEAT_COLS = D['feat_cols']
Xtr = D['train']['X_feat'].astype(np.float64); ytr = D['train']['y']
Fte = D['test']['X_feat'].astype(np.float64); yte = np.asarray(D['test']['y'])
Xseq_te, lte = T.pad(D['test']['X_tensor'])
Xseq_te_t = torch.FloatTensor(Xseq_te).to(device)
lte_t = torch.LongTensor(lte).to(device)

from sklearn.metrics import average_precision_score

# ---------- Token-Attn (seed42) ----------
ckpt = torch.load(f'{T.OUT}/c1c2_tokenattn_model.pt', map_location='cpu')
model = T.TokenAttnFusion(62).to(device)
model.load_state_dict(ckpt['state']); model.eval()
def ta_predict(F):
    with torch.no_grad():
        Ft = torch.FloatTensor(F.astype(np.float32)).to(device)
        return torch.softmax(model(Xseq_te_t, lte_t, Ft), 1)[:, 1].cpu().numpy()

# ---------- TabPFN (seed42) ----------
from tabpfn import TabPFNClassifier
clf = TabPFNClassifier(device='cuda', n_estimators='auto', balance_probabilities=True,
                       random_state=42, show_progress_bar=False)
print('fitting TabPFN...', flush=True)
clf.fit(Xtr, ytr)
def tab_predict(F):
    return clf.predict_proba(F)[:, 1]

def perm_importance(predict_fn, Fte, n_repeat=3):
    """逐特征打乱, 返回 [62] 的 PR-AUC 下降 (正值=重要)。"""
    base = average_precision_score(yte, predict_fn(Fte))
    drops = np.zeros(len(FEAT_COLS))
    rng = np.random.default_rng(0)
    for j in range(len(FEAT_COLS)):
        ds = []
        for _ in range(n_repeat):
            Fp = Fte.copy()
            rng.shuffle(Fp[:, j])
            ds.append(base - average_precision_score(yte, predict_fn(Fp)))
        drops[j] = float(np.mean(ds))
    return base, drops

print('Token-Attn permutation importance (62 feat × 3)...', flush=True)
t0 = time.time()
base_ta, drops_ta = perm_importance(ta_predict, Fte)
print(f'  base PR-AUC={base_ta:.4f}, done {time.time()-t0:.0f}s', flush=True)

print('TabPFN permutation importance (62 feat × 3)...', flush=True)
t0 = time.time()
base_tab, drops_tab = perm_importance(tab_predict, Fte)
print(f'  base PR-AUC={base_tab:.4f}, done {time.time()-t0:.0f}s', flush=True)

# ---------- 聚合到特征组 ----------
GROUPS = {
    'basic': {'v_mean','v_std','v_min','v_max','v_first','v_last','v_slope','a_mean','a_std','a_min','a_max','a_first','a_last','p_mean','p_std','p_max','p_min','p_first','p_last','soc_first','soc_last','soc_delta','t1_mean','t1_max','t2_mean','t2_max','t1_last','t2_last','n_points','duration_min','total_kwh','p_v_ratio'},
    'tempchg': {'t1_slope','t2_slope','t1_max_jump','t2_max_jump','t1_std_2nd','t2_std_2nd'},
    'segdiff': {'v_first_third_min','v_sag_from_mean','a_last_third_max','a_first_last_ratio','p_max_jump','soc_rate','soc_last_rate','v_seg_change_1to2','v_seg_change_2to3','a_seg_change_1to2','a_seg_change_2to3','a_seg3_max','v_last3_vs_first3','a_last3_vs_first3','p_last3_vs_first3'},
    'overtemp': {'t1_over_40','t2_over_40','t1_over_45','t2_over_45'},
    'batt': {'bt_LFP','bt_NMC','bt_LMO','bt_LCO','bt_LP'},
}
def group_agg(drops):
    out = {}
    for g, names in GROUPS.items():
        idx = [i for i, c in enumerate(FEAT_COLS) if c in names]
        out[g] = float(np.sum([drops[i] for i in idx]))
    return out

# 排序取 top 特征
order_ta = np.argsort(drops_ta)[::-1]
top_ta = [(FEAT_COLS[i], round(drops_ta[i], 4)) for i in order_ta[:10]]
order_tab = np.argsort(drops_tab)[::-1]
top_tab = [(FEAT_COLS[i], round(drops_tab[i], 4)) for i in order_tab[:10]]

# 相关系数
corr = float(np.corrcoef(drops_ta, drops_tab)[0, 1])

res = {
    'base_ta': base_ta, 'base_tab': base_tab,
    'per_feature': {c: {'tokenattn': float(drops_ta[i]), 'tabpfn': float(drops_tab[i])}
                    for i, c in enumerate(FEAT_COLS)},
    'group_agg': {'tokenattn': group_agg(drops_ta), 'tabpfn': group_agg(drops_tab)},
    'top10_tokenattn': top_ta, 'top10_tabpfn': top_tab,
    'corr': corr,
}
json.dump(res, open(f'{_ROOT}/r3_feature_importance.json', 'w'), indent=2, ensure_ascii=False)

# ---------- 出图: 散点对齐 ----------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
os.makedirs(f'{_ROOT}/r3_figs', exist_ok=True)
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(drops_tab, drops_ta, s=30, alpha=0.7)
lim = max(drops_ta.max(), drops_tab.max()) * 1.1
ax.plot([0, lim], [0, lim], 'k--', lw=1)
ax.set_xlabel('TabPFN permutation importance (ΔPR-AUC)')
ax.set_ylabel('Token-Attn permutation importance (ΔPR-AUC)')
ax.set_title(f'Feature attribution alignment (r={corr:.2f})')
for i in list(order_ta[:5]) + list(order_tab[:5]):
    ax.annotate(FEAT_COLS[i], (drops_tab[i], drops_ta[i]), fontsize=7, alpha=0.8)
plt.tight_layout()
plt.savefig(f'{_ROOT}/r3_figs/feature_importance_alignment.png', dpi=150, bbox_inches='tight')
plt.close()

print('\n=== R3 特征归因对齐 ===', flush=True)
print(f'  Token-Attn top5: {top_ta[:5]}', flush=True)
print(f'  TabPFN     top5: {top_tab[:5]}', flush=True)
print(f'  Spearman/线性 corr = {corr:.3f}', flush=True)
print(f'  组聚合 Token-Attn: {group_agg(drops_ta)}', flush=True)
print(f'  组聚合 TabPFN:     {group_agg(drops_tab)}', flush=True)
print('DONE', flush=True)
