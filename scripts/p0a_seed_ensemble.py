#!/usr/bin/env python3
"""P0a: 3-seed 概率集成 (seed ensemble)。

对 tokenattn 3 个 seed (42/123/2024) 的测试集概率取平均, 重算指标。
期望: PR-AUC 较单 seed 均值 0.866 有小幅提升 + 方差下降。
输出: docs/p0a_seed_ensemble.json
"""
import os, numpy as np, pickle, warnings, json
warnings.filterwarnings('ignore')
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = _ROOT + '/data/real/'
OUT = _ROOT + '/docs/'

D = pickle.load(open(f'{DATA}/fusion_data.pkl', 'rb'))
yte = np.asarray(D['test']['y'])

seeds = [42, 123, 2024, 7, 99, 500, 2025]
probs = {}
for s in seeds:
    sfix = f'_s{s}' if s != 42 else ''
    p = np.load(f'{OUT}/c1c2_tokenattn{sfix}_prob.npy')
    probs[s] = p
    print(f'  seed {s:4d}: PR-AUC={average_precision_score(yte, p):.4f}  '
          f'AUC={roc_auc_score(yte, p):.4f}  F1_05={f1_score(yte, (p >= 0.5).astype(int), zero_division=0):.4f}', flush=True)

ens = np.mean(list(probs.values()), axis=0)

# 校准 F1: 在 val 上扫阈值(与 train_c1c2 同法), 但这里 val prob 未存, 用固定 th=0.5 与简单多数
res = {
    'ensemble_PR-AUC': float(average_precision_score(yte, ens)),
    'ensemble_AUC': float(roc_auc_score(yte, ens)),
    'ensemble_F1_05': float(f1_score(yte, (ens >= 0.5).astype(int), zero_division=0)),
    'single_seed_mean_PR-AUC': float(np.mean([average_precision_score(yte, p) for p in probs.values()])),
    'single_seed_std_PR-AUC': float(np.std([average_precision_score(yte, p) for p in probs.values()])),
    'seeds': {str(s): {'PR-AUC': float(average_precision_score(yte, p))} for s, p in probs.items()},
}
print('\n=== P0a seed ensemble (owner7-8 test, 2776 序列) ===', flush=True)
for k, v in res.items():
    if k != 'seeds':
        print(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}', flush=True)
print(f'  gain over single-seed-mean: {res["ensemble_PR-AUC"] - res["single_seed_mean_PR-AUC"]:+.4f}', flush=True)

json.dump(res, open(f'{OUT}/p0a_seed_ensemble.json', 'w'), indent=2)
np.save(f'{OUT}/p0a_tokenattn_ensemble_prob.npy', ens)
print('DONE', flush=True)
