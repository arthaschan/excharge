#!/usr/bin/env python3
"""P0d: TabPFN-2.5 on 62 维特征 (旁证: 换最新表格基础模型能否超 GBDT)。

口径与 GBDT 完全一致: 62 维特征, owner1-6 训练 / owner7-8 测试, 主指标 PR-AUC。
对照: LightGBM 0.8684 / XGBoost 0.8874 (H100 实测)。
同时跑 balance_probabilities=True/False 两档, 看类别不平衡处理的影响。
输出: docs/p0d_tabpfn.json
"""
import os, pickle, warnings, json, time
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, recall_score, precision_score, accuracy_score

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = _ROOT + '/data/real/'
OUT = _ROOT + '/docs/'

D = pickle.load(open(f'{DATA}/fusion_data.pkl', 'rb'))
Xtr = D['train']['X_feat'].astype(np.float64); ytr = D['train']['y']
Xva = D['val']['X_feat'].astype(np.float64); yva = D['val']['y']
Xte = D['test']['X_feat'].astype(np.float64); yte = D['test']['y']
print(f'Train {Xtr.shape} (fault {int(ytr.sum())}) | Val {Xva.shape} | Test {Xte.shape} (fault {int(yte.sum())})', flush=True)

from tabpfn import TabPFNClassifier

def report(name, p_te, p_va):
    best_th, best_f1 = 0.5, 0
    for th in np.arange(0.05, 0.96, 0.01):
        f1 = f1_score(yva, (p_va >= th).astype(int), zero_division=0)
        if f1 > best_f1: best_f1, best_th = f1, float(th)
    cpred = (p_te >= best_th).astype(int)
    return {
        'PR-AUC': float(average_precision_score(yte, p_te)),
        'AUC': float(roc_auc_score(yte, p_te)),
        'F1_05': float(f1_score(yte, (p_te >= 0.5).astype(int), zero_division=0)),
        'Recall_05': float(recall_score(yte, (p_te >= 0.5).astype(int))),
        'best_th': best_th,
        'calib_F1': float(f1_score(yte, cpred, zero_division=0)),
        'calib_Recall': float(recall_score(yte, cpred)),
    }

results = {}
for balance in [True, False]:
    tag = 'balanced' if balance else 'default'
    print(f'\n===== TabPFN (balance_probabilities={balance}) =====', flush=True)
    clf = TabPFNClassifier(device='cuda', n_estimators='auto', balance_probabilities=balance,
                           random_state=42, show_progress_bar=False)
    t0 = time.time()
    clf.fit(Xtr, ytr)
    print(f'  fit done {time.time()-t0:.0f}s', flush=True)
    t0 = time.time()
    p_te = clf.predict_proba(Xte)[:, 1]
    p_va = clf.predict_proba(Xva)[:, 1]
    print(f'  predict done {time.time()-t0:.0f}s', flush=True)
    results[tag] = report(tag, p_te, p_va)
    np.save(f'{OUT}/p0d_tabpfn_{tag}_prob.npy', p_te)

out = {'results': results,
       'reference': {'lightgbm_PR-AUC': 0.8684, 'xgboost62_PR-AUC': 0.8874,
                     'tokenattn_7seed_ensemble_PR-AUC': 0.9184}}
json.dump(out, open(f'{OUT}/p0d_tabpfn.json', 'w'), indent=2)
print('\n=== P0d TabPFN (owner7-8 test, 62 feat) ===', flush=True)
for tag, r in results.items():
    print(f'  {tag:10s} PR-AUC={r["PR-AUC"]:.4f}  AUC={r["AUC"]:.4f}  F1_05={r["F1_05"]:.4f}  calib_F1={r["calib_F1"]:.4f} (th={r["best_th"]:.2f})', flush=True)
print('  参照: LightGBM 0.8684 | XGBoost 0.8874 | Token-Attn集成 0.9184', flush=True)
print('DONE', flush=True)
