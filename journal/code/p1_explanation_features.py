#!/usr/bin/env python3
"""P1 解释增强特征 — 方案A 第一探针（LightGBM paired 对照）

背景：老师方向"把可解释性分析结果当新特征增强加进检测模型"。
本脚本把论文 §4.3/§4.5 已实测的归因/成因机制，显式构造成 5 个"比值/速率"型新特征
（62 维里是原始统计量，没有这些比值），加进 62 维 → 67 维，
paired 对照 LightGBM(62) vs LightGBM(67)，看 PR-AUC 是否提升。

5 个解释驱动特征（每条对应一条实测机制，见 journal/docs/期刊方案v2 §3 P1）：
  power_decay_ratio  = p_last / p_max               # 功率未衰减度（末端功率占峰值 98% vs 24%）
  power_tail_ratio   = mean(p[后1/3]) / p_max       # 中后段功率维持度（注意力中后段漂移）
  soc_utilization    = soc_delta / (100 - soc_first)# 相对剩余空间充入进度（末端 SOC 84% vs 98%）
  temp_duration_rate = t2_max / duration_min        # 枪温累积速率（温度-时长伴随 r≈0.13）
  power_onset_ratio  = p_first / p_max              # 首段功率占峰值比（首段即达峰）

口径（与论文/复现包严格一致）：
  - 数据 fusion_data.pkl（62 维 z-score 特征 + tx ID + owner1-6 train / owner7-8 test，seed=42）
  - 主指标 PR-AUC；LightGBM 超参同 train_gbdt.py（n_est=1000, lr=0.05, leaves=31, early_stop=100）
  - 新特征原始计算自 all_data.parquet（按 tx 分组、按 begin_time 排序），训练域中位数填充 + z-score(fit train)

输出：journal/docs/p1_explanation_features.json
判据：67 维 PR-AUC 显著高于 62 维 → 反哺成立；否则记录诚实负结果。
"""
import pickle, numpy as np, time, json, os, warnings
import pandas as pd
warnings.filterwarnings('ignore')
import lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BASE)
DATA = os.path.join(ROOT, 'data', 'real')
OUT = os.path.join(BASE, 'docs')
os.makedirs(OUT, exist_ok=True)
SEEDS = [42, 123, 2024]
EPS = 1e-6

# ---------- 1. 载入 62 维特征（已 z-score） ----------
t0 = time.time()
D = pickle.load(open(f'{DATA}/fusion_data.pkl', 'rb'))
Ftr62 = D['train']['X_feat'].astype(np.float32); ytr = D['train']['y']; tr_tx = D['train']['tx']
Fva62 = D['val']['X_feat'].astype(np.float32);   yva = D['val']['y'];   va_tx = D['val']['tx']
Fte62 = D['test']['X_feat'].astype(np.float32);  yte = D['test']['y'];  te_tx = D['test']['tx']
print(f'[1/4] 62维已载入 | train {Ftr62.shape} val {Fva62.shape} test {Fte62.shape} | fault_test={yte.sum()}', flush=True)

# ---------- 2. 计算 5 个解释驱动特征（raw，按 tx 分组） ----------
print('[2/4] 从 all_data.parquet 计算 5 个解释驱动特征 ...', flush=True)
df = pd.read_parquet(f'{DATA}/all_data.parquet')
grp = {tx: sub.sort_values('begin_time') for tx, sub in df.groupby('transaction_id', sort=False)}
print(f'  all_data {len(df):,} 行 | 事务 {len(grp):,}', flush=True)

def xai_features(sub):
    p = sub['out_power'].to_numpy(dtype=np.float64)
    soc = sub['current_soc'].to_numpy(dtype=np.float64)
    t2 = sub['charging_gun_temperature2'].to_numpy(dtype=np.float64)
    dur = float(sub['total_charging_min'].max())
    n = len(p)
    f = {}
    pmax = float(np.nanmax(p)) if n else np.nan
    f['power_decay_ratio'] = (float(p[-1]) / pmax) if (n and np.isfinite(pmax) and pmax > EPS) else np.nan
    tail = p[max(0, 2 * n // 3):]
    f['power_tail_ratio'] = (float(np.nanmean(tail)) / pmax) if (len(tail) and np.isfinite(pmax) and pmax > EPS) else np.nan
    s0 = float(soc[0]) if (n and np.isfinite(soc[0])) else np.nan
    s1 = float(soc[-1]) if (n and np.isfinite(soc[-1])) else np.nan
    denom = 100.0 - s0
    f['soc_utilization'] = ((s1 - s0) / denom) if (np.isfinite(s0) and np.isfinite(s1) and abs(denom) > EPS) else np.nan
    t2max = float(np.nanmax(t2)) if n else np.nan
    f['temp_duration_rate'] = (t2max / dur) if (np.isfinite(t2max) and dur > EPS) else np.nan
    f['power_onset_ratio'] = (float(p[0]) / pmax) if (n and np.isfinite(pmax) and pmax > EPS) else np.nan
    return f

XAI_COLS = ['power_decay_ratio', 'power_tail_ratio', 'soc_utilization', 'temp_duration_rate', 'power_onset_ratio']

def assemble(tx_list):
    M = np.zeros((len(tx_list), len(XAI_COLS)), dtype=np.float64)
    miss = 0
    for i, tx in enumerate(tx_list):
        if tx not in grp:
            M[i, :] = np.nan; miss += 1; continue
        f = xai_features(grp[tx])
        for j, c in enumerate(XAI_COLS):
            M[i, j] = f.get(c, np.nan)
    return M, miss

Xtr_x, m1 = assemble(tr_tx)
Xva_x, m2 = assemble(va_tx)
Xte_x, m3 = assemble(te_tx)
print(f'  未命中 tx: train {m1} / val {m2} / test {m3}', flush=True)

# 训练域中位数填充 + z-score(fit train)，与 62 维同处理
med = np.nanmedian(Xtr_x, axis=0)
Xtr_x = np.where(np.isnan(Xtr_x), med, Xtr_x).astype(np.float32)
Xva_x = np.where(np.isnan(Xva_x), med, Xva_x).astype(np.float32)
Xte_x = np.where(np.isnan(Xte_x), med, Xte_x).astype(np.float32)
mu = Xtr_x.mean(axis=0); sd = Xtr_x.std(axis=0); sd = np.where(sd < 1e-8, 1.0, sd)
Xtr_x = (Xtr_x - mu) / sd; Xva_x = (Xva_x - mu) / sd; Xte_x = (Xte_x - mu) / sd
print(f'  新特征矩阵: train {Xtr_x.shape} (z-scored)', flush=True)

# 拼接 67 维
Ftr67 = np.hstack([Ftr62, Xtr_x]); Fva67 = np.hstack([Fva62, Xva_x]); Fte67 = np.hstack([Fte62, Xte_x])
print(f'[3/4] 67维已拼接: {Ftr67.shape} / {Fva67.shape} / {Fte67.shape}', flush=True)

# ---------- 3. LightGBM paired 对照 ----------
def train_lgb(Ftr, Fva, ytr, yva, seed):
    m = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=31,
                           subsample=0.8, colsample_bytree=0.8, random_state=seed,
                           n_jobs=4, verbosity=-1)
    m.fit(Ftr, ytr, eval_set=[(Fva, yva)], eval_metric='auc',
          callbacks=[lgb.early_stopping(100, verbose=False)])
    return m

def paired_bootstrap(yte, p_a, p_b, n_boot=10000, seed=0):
    """p_a=62维 prob, p_b=67维 prob。返回 (diff_mean, p_better=P(PR-AUC67<=PR-AUC62))。"""
    rng = np.random.default_rng(seed)
    n = len(yte)
    d = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        d[i] = average_precision_score(yte[idx], p_b[idx]) - average_precision_score(yte[idx], p_a[idx])
    return float(d.mean()), float((d <= 0).mean())

results = {'meta': {'date': '2026-09-04', 'env': 'H100', 'seeds': SEEDS,
                    'protocol': 'owner1-6 train / owner7-8 test, seed42 划分, 与论文同口径',
                    'xai_features': XAI_COLS,
                    'xai_rationale': '功率未衰减/中后段漂移/SOC未满/温度-时长/首段即达峰 —— 论文§4.3/§4.5实测机制'},
           'per_seed': {}, 'headline': {}}

for sd in SEEDS:
    t1 = time.time()
    m62 = train_lgb(Ftr62, Fva62, ytr, yva, sd)
    m67 = train_lgb(Ftr67, Fva67, ytr, yva, sd)
    p62 = m62.predict_proba(Fte62)[:, 1]; p67 = m67.predict_proba(Fte67)[:, 1]
    r = {
        'pr62': float(average_precision_score(yte, p62)), 'auc62': float(roc_auc_score(yte, p62)),
        'pr67': float(average_precision_score(yte, p67)), 'auc67': float(roc_auc_score(yte, p67)),
        'delta_pr': float(average_precision_score(yte, p67) - average_precision_score(yte, p62)),
        'best_iter62': int(m62.best_iteration_), 'best_iter67': int(m67.best_iteration_),
        'sec': round(time.time() - t1, 1),
    }
    results['per_seed'][str(sd)] = r
    print(f'  seed={sd}: 62维 PR-AUC={r["pr62"]:.4f} | 67维 PR-AUC={r["pr67"]:.4f} | Δ={r["delta_pr"]:+.4f} '
          f'(AUC62={r["auc62"]:.4f}→AUC67={r["auc67"]:.4f})', flush=True)

# 显著性（seed=42 与论文基线同 seed）
prs = [results['per_seed'][str(s)]['pr62'] for s in SEEDS]
prs67 = [results['per_seed'][str(s)]['pr67'] for s in SEEDS]
deltas = [results['per_seed'][str(s)]['delta_pr'] for s in SEEDS]

# 重训 seed=42 拿 prob 做 bootstrap（上面已算过 p62/p67，这里再单独跑一次存 prob）
m62 = train_lgb(Ftr62, Fva62, ytr, yva, 42); m67 = train_lgb(Ftr67, Fva67, ytr, yva, 42)
p62 = m62.predict_proba(Fte62)[:, 1]; p67 = m67.predict_proba(Fte67)[:, 1]
diff_mean, p_better = paired_bootstrap(yte, p62, p67)
np.save(f'{OUT}/p1_lightgbm_62_prob.npy', p62)
np.save(f'{OUT}/p1_lightgbm_67_prob.npy', p67)

results['headline'] = {
    'pr62_mean': float(np.mean(prs)), 'pr62_std': float(np.std(prs)),
    'pr67_mean': float(np.mean(prs67)), 'pr67_std': float(np.std(prs67)),
    'delta_pr_mean': float(np.mean(deltas)), 'delta_pr_std': float(np.std(deltas)),
    'bootstrap_diff_mean(seed42)': diff_mean,
    'bootstrap_p_better(67>62)': 1.0 - p_better,
    'bootstrap_p_worse_or_equal': p_better,
}

with open(f'{OUT}/p1_explanation_features.json', 'w', encoding='utf-8') as fp:
    json.dump(results, fp, ensure_ascii=False, indent=2)

print('[4/4] 结果已保存 journal/docs/p1_explanation_features.json', flush=True)
print('\n=== P1 判据 ===', flush=True)
h = results['headline']
print(f'  62维 PR-AUC = {h["pr62_mean"]:.4f}±{h["pr62_std"]:.4f} (3 seed)', flush=True)
print(f'  67维 PR-AUC = {h["pr67_mean"]:.4f}±{h["pr67_std"]:.4f} (3 seed)', flush=True)
print(f'  Δ PR-AUC = {h["delta_pr_mean"]:+.4f}±{h["delta_pr_std"]:.4f}', flush=True)
print(f'  bootstrap: diff_mean={h["bootstrap_diff_mean(seed42)"]:+.4f}, p(67>62)={h["bootstrap_p_better(67>62)"]:.4f}, p(67<=62)={h["bootstrap_p_worse_or_equal"]:.4f}', flush=True)
print(f'  总耗时 {time.time()-t0:.0f}s', flush=True)
