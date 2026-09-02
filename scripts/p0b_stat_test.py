#!/usr/bin/env python3
"""P0b: 统计检验 tokenattn(3-seed 集成) vs LightGBM 的性能差异。

方法 1 (主): 配对 bootstrap —— 对 owner7-8 测试集(2776)有放回重采样 B=10000 次,
    每次算两个模型 PR-AUC 的差, 得到差的分布 → 95% CI + 双侧 p 值。
方法 2 (辅): DeLong 检验 (AUC), 经典结构化分量法 (DeLong et al. 1988)。
输出: docs/p0b_stat_test.json
"""
import os, numpy as np, pickle, warnings, json
warnings.filterwarnings('ignore')
from sklearn.metrics import average_precision_score, roc_auc_score

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = _ROOT + '/data/real/'
OUT = _ROOT + '/docs/'
B = 10000
rng = np.random.default_rng(42)

D = pickle.load(open(f'{DATA}/fusion_data.pkl', 'rb'))
yte = np.asarray(D['test']['y'])

# tokenattn 3-seed 集成概率
seeds = [42, 123, 2024, 7, 99, 500, 2025]
probs = []
for s in seeds:
    sfix = f'_s{s}' if s != 42 else ''
    probs.append(np.load(f'{OUT}/c1c2_tokenattn{sfix}_prob.npy'))
p_ta = np.mean(probs, axis=0)

# LightGBM 概率
p_lgb = np.load(f'{OUT}/gbdt_lightgbm_prob.npy')

prauc_ta = average_precision_score(yte, p_ta)
prauc_lgb = average_precision_score(yte, p_lgb)
auc_ta = roc_auc_score(yte, p_ta)
auc_lgb = roc_auc_score(yte, p_lgb)
print(f'  tokenattn-ens  PR-AUC={prauc_ta:.4f}  AUC={auc_ta:.4f}', flush=True)
print(f'  lightgbm       PR-AUC={prauc_lgb:.4f}  AUC={auc_lgb:.4f}', flush=True)

# ---------- 配对 bootstrap (PR-AUC) ----------
n = len(yte)
diffs = np.zeros(B)
for b in range(B):
    idx = rng.integers(0, n, n)
    yb, a_b, g_b = yte[idx], p_ta[idx], p_lgb[idx]
    if yb.sum() == 0 or yb.sum() == n:
        diffs[b] = 0.0
        continue
    diffs[b] = average_precision_score(yb, a_b) - average_precision_score(yb, g_b)

lo, hi = np.percentile(diffs, [2.5, 97.5])
# 双侧 p 值: 差分布在 0 同侧的比例 × 2
p_boot = 2 * min((diffs > 0).mean(), (diffs < 0).mean())
if (diffs == 0).any():
    p_boot = min(1.0, p_boot + (diffs == 0).mean())

# ---------- DeLong (AUC) ----------
def delong_structs(scores):
    pos = scores[yte == 1]; neg = scores[yte == 0]
    n1, n0 = len(pos), len(neg)
    # psi(a,b)=1 if a>b, .5 if ==, 0 else
    V10 = np.array([(np.sum(p > neg) + 0.5 * np.sum(p == neg)) / n0 for p in pos])  # n1
    V01 = np.array([(np.sum(pos > g) + 0.5 * np.sum(pos == g)) / n1 for g in neg])  # n0
    return V10, V01, n1, n0

V10a, V01a, n1, n0 = delong_structs(p_ta)
V10g, V01g, _, _ = delong_structs(p_lgb)

def cov_term(Va, Vb, n):
    m = np.stack([Va, Vb], 1)          # [n,2]
    return np.cov(m, rowvar=False, bias=True) / n   # [2,2]

S10 = cov_term(V10a, V10g, n1)
S01 = cov_term(V01a, V01g, n0)
Cov = S10 + S01                          # [2,2] cov of (AUC_a, AUC_g)
var_diff = Cov[0, 0] + Cov[1, 1] - 2 * Cov[0, 1]
z = (auc_ta - auc_lgb) / np.sqrt(var_diff) if var_diff > 0 else 0.0
from scipy import stats
p_delong = 2 * stats.norm.sf(abs(z))

res = {
    'tokenattn_ensemble': {'PR-AUC': float(prauc_ta), 'AUC': float(auc_ta)},
    'lightgbm': {'PR-AUC': float(prauc_lgb), 'AUC': float(auc_lgb)},
    'PR-AUC_diff_ta_minus_lgb': float(prauc_ta - prauc_lgb),
    'bootstrap': {'B': B, 'diff_mean': float(diffs.mean()), 'diff_std': float(diffs.std()),
                  'CI95': [float(lo), float(hi)], 'p_two_sided': float(p_boot)},
    'delong_AUC': {'z': float(z), 'p_two_sided': float(p_delong)},
}
print('\n=== P0b 统计检验 ===', flush=True)
print(f'  bootstrap: diff={diffs.mean():+.4f} ± {diffs.std():.4f}  95%CI=[{lo:+.4f},{hi:+.4f}]  p={p_boot:.4f}', flush=True)
print(f'  DeLong(AUC): z={z:.3f}  p={p_delong:.4f}', flush=True)
print(f'  结论: {"无显著差异(打平成立)" if p_boot > 0.05 else "存在显著差异"}', flush=True)
json.dump(res, open(f'{OUT}/p0b_stat_test.json', 'w'), indent=2)
print('DONE', flush=True)
