#!/usr/bin/env python3
"""Train enhanced XGBoost on v2 features, compare against Route B baseline.
Target: improve Recall beyond 58.14% (esp. on FN overtemperature faults).
"""
import pandas as pd, numpy as np, json, time, warnings
warnings.filterwarnings('ignore')
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

DATA = '/Users/arthas/git/excharge/data/real/'
OUT  = '/Users/arthas/git/excharge/docs/'

X_train = pd.read_parquet(f'{DATA}/seq_X_train_v2.parquet')
X_test  = pd.read_parquet(f'{DATA}/seq_X_test_v2.parquet')
y_train = pd.read_parquet(f'{DATA}/seq_y_train_v2.parquet')['label'].values
y_test  = pd.read_parquet(f'{DATA}/seq_y_test_v2.parquet')['label'].values

print(f'Train {len(X_train):,} (fault {y_train.sum():,}) | Test {len(X_test):,} (fault {y_test.sum():,})')

# ---- Train ----
spw = (y_train == 0).sum() / (y_train == 1).sum()
t0 = time.time()
xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                    scale_pos_weight=spw, eval_metric='logloss',
                    random_state=42, n_jobs=-1)
xgb.fit(X_train, y_train, verbose=False)
print(f'Trained in {time.time()-t0:.1f}s')

pred = xgb.predict(X_test)
prob = xgb.predict_proba(X_test)[:, 1]

acc  = accuracy_score(y_test, pred)
prec = precision_score(y_test, pred, zero_division=0)
rec  = recall_score(y_test, pred)
f1   = f1_score(y_test, pred)
auc  = roc_auc_score(y_test, prob)
print(f'\n=== ENHANCED (v2) Results ===')
print(f'Acc={acc:.4f} Prec={prec:.4f} Recall={rec:.4f} F1={f1:.4f} AUC={auc:.4f}')

tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
print(f'Confusion: TP={tp} FN={fn} FP={fp} TN={tn}')

# ---- Feature importance ----
imp = pd.Series(xgb.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print('\n=== Top20 Feature Importance ===')
for f, v in imp.head(20).items():
    print(f'  {f}: {v:.4f}')

# ---- Which faults now recovered? ----
old_pred = np.load(f'{OUT}/routeA_test_pred.npy')
new_recovered = (old_pred == 0) & (y_test == 1) & (pred == 1)
still_missed  = (pred == 0) & (y_test == 1)
print(f'\n=== FN Recovery ===')
print(f'Old FN: {int(((old_pred==0)&(y_test==1)).sum())} | New FN: {int(still_missed.sum())} | Recovered: {int(new_recovered.sum())}')

# ---- SHAP on v2 ----
import shap
print('\nComputing SHAP...', flush=True)
t0 = time.time()
explainer = shap.TreeExplainer(xgb)
shap_vals = explainer.shap_values(X_test)
print(f'SHAP done in {time.time()-t0:.0f}s')
fi_shap = pd.Series(np.abs(shap_vals).mean(axis=0), index=X_train.columns).sort_values(ascending=False)
print('\n=== Top15 SHAP (v2) ===')
for f, v in fi_shap.head(15).items():
    print(f'  {f}: {v:.4f}')

# FN vs TP SHAP on new features
fn_mask = still_missed
tp_mask = (pred == 1) & (y_test == 1)
if fn_mask.sum() > 0:
    fn_shap = pd.Series(np.abs(shap_vals[fn_mask]).mean(axis=0), index=X_train.columns).sort_values(ascending=False)
    tp_shap = pd.Series(np.abs(shap_vals[tp_mask]).mean(axis=0), index=X_train.columns).sort_values(ascending=False)
    print('\n=== FN samples: top10 SHAP ===')
    for f, v in fn_shap.head(10).items():
        print(f'  {f}: {v:.4f}')
    print('\n=== TP samples: top10 SHAP ===')
    for f, v in tp_shap.head(10).items():
        print(f'  {f}: {v:.4f}')
    fn_tp = pd.concat([fn_shap.rename('fn'), tp_shap.rename('tp')], axis=1)
    fn_tp['diff'] = fn_tp['fn'] - fn_tp['tp']
    fn_tp_sorted = fn_tp.sort_values('diff', ascending=False)
    print('\n=== Features FN>T TTP (why still missed) ===')
    for f, row in fn_tp_sorted.head(10).iterrows():
        print(f'  {f}: FN={row["fn"]:.4f} TP={row["tp"]:.4f} diff={row["diff"]:+.4f}')

# ---- Save ----
np.save(f'{OUT}/routeA_v2_pred.npy', pred)
np.save(f'{OUT}/routeA_v2_prob.npy', prob)
np.save(f'{OUT}/routeA_v2_shap.npy', shap_vals)
results = {
    'model': 'XGBoost-enhanced',
    'metrics': {'acc': acc, 'prec': prec, 'recall': rec, 'f1': f1, 'auc': auc},
    'confusion': {'tp': int(tp), 'fn': int(fn), 'fp': int(fp), 'tn': int(tn)},
    'old_recall': 0.5814,
    'recall_gain': rec - 0.5814,
    'recovered_fn': int(new_recovered.sum()),
    'n_features': int(X_train.shape[1]),
    'top15_shap': fi_shap.head(15).to_dict(),
}
json.dump(results, open(f'{OUT}/routeA_v2_results.json', 'w'), indent=2, default=str)
print('\nSaved routeA_v2_results.json')
