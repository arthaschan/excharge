#!/usr/bin/env python3
"""P0d 补充: TabPFN-2.5 多 seed 复现 + 概率校准 (R3 证据)。

1) 3 个 seed (42/123/2024) 各跑一次 TabPFN-2.5, 报 PR-AUC 均值±std;
2) 3-seed 概率集成;
3) Platt 缩放校准(在 val 上拟合), 报校准前后 F1 + ECE/Brier(应对"过度自信"隐患)。
输出: docs/p0d_tabpfn_multiseed.json + docs/p0d_tabpfn_ensemble_prob.npy
"""
import os, pickle, warnings, json, time
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, recall_score, precision_score

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = _ROOT + '/data/real/'
OUT = _ROOT + '/docs/'

D = pickle.load(open(f'{DATA}/fusion_data.pkl', 'rb'))
Xtr = D['train']['X_feat'].astype(np.float64); ytr = D['train']['y']
Xva = D['val']['X_feat'].astype(np.float64); yva = D['val']['y']
Xte = D['test']['X_feat'].astype(np.float64); yte = D['test']['y']
print(f'Train {Xtr.shape} (fault {int(ytr.sum())}) | Val {Xva.shape} | Test {Xte.shape} (fault {int(yte.sum())})', flush=True)

from tabpfn import TabPFNClassifier

SEEDS = [42, 123, 2024]
probs = {}; valprobs = {}
per_seed = []
for s in SEEDS:
    print(f'--- seed {s} ---', flush=True)
    clf = TabPFNClassifier(device='cuda', n_estimators='auto', balance_probabilities=True,
                           random_state=s, show_progress_bar=False)
    t0 = time.time()
    clf.fit(Xtr, ytr)
    p_te = clf.predict_proba(Xte)[:, 1]
    p_va = clf.predict_proba(Xva)[:, 1]
    prauc = float(average_precision_score(yte, p_te))
    probs[s] = p_te; valprobs[s] = p_va
    per_seed.append(prauc)
    print(f'  seed {s}: PR-AUC={prauc:.4f} ({time.time()-t0:.0f}s)', flush=True)
    np.save(f'{OUT}/p0d_tabpfn_s{s}_prob.npy', p_te)

ens = np.mean(list(probs.values()), axis=0)
ens_va = np.mean(list(valprobs.values()), axis=0)
ens_prauc = float(average_precision_score(yte, ens))
mean_prauc = float(np.mean(per_seed)); std_prauc = float(np.std(per_seed))

# ---------- Platt 缩放校准 (在 val 上拟合) ----------
def platt_calibrate(p_va, p_te, yva):
    from sklearn.linear_model import LogisticRegression
    logit_va = np.log(np.clip(p_va, 1e-12, 1 - 1e-12) / np.clip(1 - p_va, 1e-12, 1 - 1e-12))
    logit_te = np.log(np.clip(p_te, 1e-12, 1 - 1e-12) / np.clip(1 - p_te, 1e-12, 1 - 1e-12))
    lr = LogisticRegression().fit(logit_va.reshape(-1, 1), yva)
    return lr.predict_proba(logit_te.reshape(-1, 1))[:, 1]

def ece(y, p, nbins=10):
    bins = np.linspace(0, 1, nbins + 1)
    idx = np.digitize(p, bins) - 1
    idx = np.clip(idx, 0, nbins - 1)
    e = 0.0
    for b in range(nbins):
        m = idx == b
        if m.sum() == 0:
            continue
        e += (m.sum() / len(y)) * abs(p[m].mean() - y[m].mean())
    return float(e)

def brier(y, p):
    return float(np.mean((p - y) ** 2))

# 用 3-seed 集成的 val 概率做 Platt 缩放校准
calib_te = platt_calibrate(ens_va, ens, yva)

# 阈值扫描(在集成 val prob 上) 得到校准前的最佳阈值 F1
best_th, best_f1 = 0.5, 0
for th in np.arange(0.05, 0.96, 0.01):
    f1 = f1_score(yva, (ens_va >= th).astype(int), zero_division=0)
    if f1 > best_f1: best_f1, best_th = f1, float(th)

res = {
    'per_seed_PR-AUC': {str(s): v for s, v in zip(SEEDS, per_seed)},
    'mean_PR-AUC': mean_prauc, 'std_PR-AUC': std_prauc,
    'ensemble_PR-AUC': ens_prauc,
    'ensemble_AUC': float(roc_auc_score(yte, ens)),
    'ensemble_F1_05': float(f1_score(yte, (ens >= 0.5).astype(int), zero_division=0)),
    'calibration': {
        'before': {'ECE': ece(yte, ens), 'Brier': brier(yte, ens)},
        'after_platt': {'ECE': ece(yte, calib_te), 'Brier': brier(yte, calib_te)},
        'calib_F1_05': float(f1_score(yte, (calib_te >= 0.5).astype(int), zero_division=0)),
        'calib_Recall_05': float(recall_score(yte, (calib_te >= 0.5).astype(int))),
    },
    'reference': {'lightgbm': 0.8684, 'xgboost62': 0.8874, 'tokenattn_7seed_ensemble': 0.9184},
}
json.dump(res, open(f'{OUT}/p0d_tabpfn_multiseed.json', 'w'), indent=2)
np.save(f'{OUT}/p0d_tabpfn_ensemble_prob.npy', ens)
np.save(f'{OUT}/p0d_tabpfn_ensemble_calibrated_prob.npy', calib_te)

print('\n=== P0d TabPFN 多 seed 复现 ===', flush=True)
for s, v in zip(SEEDS, per_seed):
    print(f'  seed {s}: PR-AUC={v:.4f}', flush=True)
print(f'  mean ± std = {mean_prauc:.4f} ± {std_prauc:.4f}', flush=True)
print(f'  ensemble PR-AUC = {ens_prauc:.4f}', flush=True)
print(f'  校准: ECE {res["calibration"]["before"]["ECE"]:.4f} -> {res["calibration"]["after_platt"]["ECE"]:.4f}; '
      f'Brier {res["calibration"]["before"]["Brier"]:.4f} -> {res["calibration"]["after_platt"]["Brier"]:.4f}', flush=True)
print('DONE', flush=True)
