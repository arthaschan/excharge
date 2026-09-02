#!/usr/bin/env python3
"""实验C：GBDT 家族对比 (XGBoost vs LightGBM)。

用 62 维特征跑 LightGBM, 与已有 XGBoost(57特征 PR-AUC 0.894 / 62特征) 对比,
补全 GBDT 家族天花板证据。同 owner7-8 测试口径。

注意: 树模型对特征 z-score 归一化不敏感, 但为与 NN 同口径仍用 X_feat(已 z-score)。
输出: docs/gbdt_compare.json
"""
import pickle, numpy as np, time, warnings, json
warnings.filterwarnings('ignore')
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_score, recall_score, accuracy_score

DATA = '/Users/arthas/git/excharge/data/real/'
OUT = '/Users/arthas/git/excharge/docs/'

with open(f'{DATA}/fusion_data.pkl', 'rb') as f:
    D = pickle.load(f)
Ftr = D['train']['X_feat'].astype(np.float32); ytr = D['train']['y']
Fva = D['val']['X_feat'].astype(np.float32);   yva = D['val']['y']
Fte = D['test']['X_feat'].astype(np.float32);  yte = D['test']['y']
print(f'Train {Ftr.shape} | Val {Fva.shape} | Test {Fte.shape} | fault_test={yte.sum()}', flush=True)

def report(name, prob):
    prauc = average_precision_score(yte, prob); auc = roc_auc_score(yte, prob)
    # 阈值校准: val 上最大化 F1
    best_th, best_f1 = 0.5, 0
    vprob = prob_val if name != 'xgb' else None
    return prauc, auc

results = {}

# ---- LightGBM ----
t0 = time.time()
m_lgb = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=31,
                           subsample=0.8, colsample_bytree=0.8, random_state=42,
                           n_jobs=4, verbosity=-1)
m_lgb.fit(Ftr, ytr, eval_set=[(Fva, yva)], eval_metric='auc',
          callbacks=[lgb.early_stopping(100, verbose=False)])
p_lgb = m_lgb.predict_proba(Fte)[:, 1]
p_lgb_va = m_lgb.predict_proba(Fva)[:, 1]
results['lightgbm'] = {
    'PR-AUC': average_precision_score(yte, p_lgb),
    'AUC': roc_auc_score(yte, p_lgb),
    'best_iter': m_lgb.best_iteration_,
    'sec': round(time.time() - t0, 1),
}

# ---- XGBoost (62 特征, 与 LightGBM 同口径) ----
t0 = time.time()
m_xgb = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=6,
                          subsample=0.8, colsample_bytree=0.8, random_state=42,
                          n_jobs=4, eval_metric='auc', early_stopping_rounds=100,
                          tree_method='hist')
m_xgb.fit(Ftr, ytr, eval_set=[(Fva, yva)], verbose=False)
p_xgb = m_xgb.predict_proba(Fte)[:, 1]
p_xgb_va = m_xgb.predict_proba(Fva)[:, 1]
results['xgboost_62feat'] = {
    'PR-AUC': average_precision_score(yte, p_xgb),
    'AUC': roc_auc_score(yte, p_xgb),
    'best_iter': m_xgb.best_iteration,
    'sec': round(time.time() - t0, 1),
}

# ---- 阈值校准(在 val 上最大化 F1) ----
def calibrate(vprob):
    best_th, best_f1 = 0.5, 0
    for th in np.arange(0.05, 0.96, 0.01):
        f1 = f1_score(yva, (vprob >= th).astype(int), zero_division=0)
        if f1 > best_f1: best_f1, best_th = f1, float(th)
    return best_th, best_f1

for name, (prob, vprob) in [('lightgbm', (p_lgb, p_lgb_va)), ('xgboost_62feat', (p_xgb, p_xgb_va))]:
    th, cf1 = calibrate(vprob)
    cpred = (prob >= th).astype(int)
    results[name]['best_th'] = th
    results[name]['calib_F1'] = f1_score(yte, cpred, zero_division=0)
    results[name]['F1_05'] = f1_score(yte, (prob >= 0.5).astype(int), zero_division=0)

json.dump(results, open(f'{OUT}/gbdt_compare.json', 'w'), indent=2)
print('\n=== GBDT 家族 (62 特征, owner7-8 测试) ===', flush=True)
for k, v in results.items():
    print(f'  {k:16s} PR-AUC={v["PR-AUC"]:.4f}  AUC={v["AUC"]:.4f}  F1_05={v["F1_05"]:.4f}  calib_F1={v["calib_F1"]:.4f} (th={v["best_th"]:.2f})', flush=True)
print('DONE', flush=True)
