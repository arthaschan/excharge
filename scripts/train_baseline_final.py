#!/usr/bin/env python3
"""Route B final: XGBoost / RF / MLP baselines on real dataset, save all results.
(Transformer windowed baseline already run separately: train_transformer_real.py)
"""
import pandas as pd, numpy as np, json, time, warnings
warnings.filterwarnings('ignore')

DATA = '/Users/arthas/git/excharge/data/real/'
OUT  = '/Users/arthas/git/excharge/docs/'

X_train = pd.read_parquet(f'{DATA}/seq_X_train.parquet')
X_val   = pd.read_parquet(f'{DATA}/seq_X_val.parquet')
X_test  = pd.read_parquet(f'{DATA}/seq_X_test.parquet')
y_train = pd.read_parquet(f'{DATA}/seq_y_train.parquet')['label'].values
y_val   = pd.read_parquet(f'{DATA}/seq_y_val.parquet')['label'].values
y_test  = pd.read_parquet(f'{DATA}/seq_y_test.parquet')['label'].values
feat_cols = list(X_train.columns)

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

results = {}

# XGBoost
from xgboost import XGBClassifier
t0 = time.time()
xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                    scale_pos_weight=(y_train==0).sum()/(y_train==1).sum(),
                    eval_metric='logloss', random_state=42, n_jobs=-1)
xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
p = xgb.predict(X_test); prob = xgb.predict_proba(X_test)[:,1]
results['XGBoost'] = {'Acc': accuracy_score(y_test,p), 'Prec': precision_score(y_test,p,zero_division=0),
                      'Recall': recall_score(y_test,p), 'F1': f1_score(y_test,p), 'AUC': roc_auc_score(y_test,prob),
                      'time_s': time.time()-t0}
fi = pd.Series(xgb.feature_importances_, index=feat_cols).sort_values(ascending=False)
print(f'XGBoost: {results["XGBoost"]}')

# RandomForest
from sklearn.ensemble import RandomForestClassifier
t0 = time.time()
rf = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
p = rf.predict(X_test); prob = rf.predict_proba(X_test)[:,1]
results['RandomForest'] = {'Acc': accuracy_score(y_test,p), 'Prec': precision_score(y_test,p,zero_division=0),
                           'Recall': recall_score(y_test,p), 'F1': f1_score(y_test,p), 'AUC': roc_auc_score(y_test,prob),
                           'time_s': time.time()-t0}
print(f'RandomForest: {results["RandomForest"]}')

# MLP
from sklearn.neural_network import MLPClassifier
t0 = time.time()
mlp = MLPClassifier(hidden_layer_sizes=(256,128,64), max_iter=300, early_stopping=True,
                    validation_fraction=0.1, random_state=42)
mlp.fit(X_train_s, y_train)
p = mlp.predict(X_test_s); prob = mlp.predict_proba(X_test_s)[:,1]
results['MLP'] = {'Acc': accuracy_score(y_test,p), 'Prec': precision_score(y_test,p,zero_division=0),
                  'Recall': recall_score(y_test,p), 'F1': f1_score(y_test,p), 'AUC': roc_auc_score(y_test,prob),
                  'time_s': time.time()-t0}
print(f'MLP: {results["MLP"]}')

# Add Transformer (from separate run)
try:
    tr = json.load(open(f'{OUT}/routeB_transformer_results.json'))
    results['Transformer (windowed)'] = tr
    print(f'Transformer: {tr}')
except Exception as e:
    print('Transformer results not found:', e)

# Save
df_res = pd.DataFrame(results).T
df_res.to_csv(f'{OUT}/routeB_baseline_results.csv')
json.dump({k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()},
          open(f'{OUT}/routeB_baseline_results.json','w'), indent=2)

# Feature importance
fi.head(15).to_frame('importance').to_csv(f'{OUT}/routeB_xgb_feature_importance.csv')

print('\n=== Model Comparison ===')
print(f'{"Model":<24} {"Acc":>8} {"Prec":>8} {"Recall":>8} {"F1":>8} {"AUC":>8} {"Time(s)":>8}')
for name, r in sorted(results.items(), key=lambda x: -x[1]['F1']):
    ts = r.get('time_s', 0)
    print(f'{name:<24} {r["Acc"]:>8.4f} {r["Prec"]:>8.4f} {r["Recall"]:>8.4f} {r["F1"]:>8.4f} {r["AUC"]:>8.4f} {ts:>8.1f}')

print('\n=== vs Nature Paper (Recall 73.56%) ===')
for name, r in results.items():
    print(f'{name}: {r["Recall"]*100:.2f}% (gap {r["Recall"]*100-73.56:+.2f}pp)')

print('\nTop15 XGBoost features:')
for f, v in fi.head(15).items():
    print(f'  {f}: {v:.4f}')
