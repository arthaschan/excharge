#!/usr/bin/env python3
"""Route A: Explainable fault detection on REAL charging data.
- Train XGBoost on real Nature dataset (seq-level features)
- TreeSHAP attribution per sample
- Fault type analysis: what features drive fault detection
- Visualization: summary plot, per-fault-type top features
"""
import pandas as pd, numpy as np, json, time, warnings, os
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap

DATA = '/Users/arthas/git/excharge/data/real/'
OUT  = '/Users/arthas/git/excharge/docs/'
os.makedirs(OUT, exist_ok=True)

# ---------- Load ----------
X_train = pd.read_parquet(f'{DATA}/seq_X_train.parquet')
X_val   = pd.read_parquet(f'{DATA}/seq_X_val.parquet')
X_test  = pd.read_parquet(f'{DATA}/seq_X_test.parquet')
y_train = pd.read_parquet(f'{DATA}/seq_y_train.parquet')['label'].values
y_val   = pd.read_parquet(f'{DATA}/seq_y_val.parquet')['label'].values
y_test  = pd.read_parquet(f'{DATA}/seq_y_test.parquet')['label'].values
feat_cols = list(X_train.columns)
print(f'Train {len(X_train):,} | Val {len(X_val):,} | Test {len(X_test):,}')
print(f'Test fault: {y_test.sum()} / {len(y_test)}')

# ---------- Train XGBoost ----------
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

t0 = time.time()
xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                    scale_pos_weight=(y_train==0).sum()/(y_train==1).sum(),
                    eval_metric='logloss', random_state=42, n_jobs=-1)
xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
print(f'XGBoost trained in {time.time()-t0:.1f}s')

pred = xgb.predict(X_test)
prob = xgb.predict_proba(X_test)[:,1]
print(f'Test: Acc={accuracy_score(y_test,pred):.4f} Prec={precision_score(y_test,pred,zero_division=0):.4f} '
      f'Recall={recall_score(y_test,pred):.4f} F1={f1_score(y_test,pred):.4f} AUC={roc_auc_score(y_test,prob):.4f}')

# ---------- TreeSHAP ----------
print('\nComputing TreeSHAP on test set (background = train)...', flush=True)
t0 = time.time()
explainer = shap.TreeExplainer(xgb)
# Use a sample of test for attribution (speed); full test for aggregate
shap_values_test = explainer.shap_values(X_test)
print(f'SHAP computed in {time.time()-t0:.0f}s, shape={shap_values_test.shape}')

# ---------- Global attribution ----------
mean_abs_shap = np.abs(shap_values_test).mean(axis=0)
fi_shap = pd.Series(mean_abs_shap, index=feat_cols).sort_values(ascending=False)
print('\n=== Top15 SHAP features (mean |SHAP|) ===')
for f, v in fi_shap.head(15).items():
    print(f'  {f}: {v:.4f}')

# ---------- Fault vs normal attribution ----------
fault_idx = np.where(y_test == 1)[0]
norm_idx  = np.where(y_test == 0)[0]
print(f'\nFault samples: {len(fault_idx)}, Normal: {len(norm_idx)}')

fault_shap = np.abs(shap_values_test[fault_idx]).mean(axis=0)
norm_shap  = np.abs(shap_values_test[norm_idx]).mean(axis=0)
fi_fault = pd.Series(fault_shap, index=feat_cols).sort_values(ascending=False)
fi_norm  = pd.Series(norm_shap, index=feat_cols).sort_values(ascending=False)

print('\n=== Top10 SHAP features: FAULT samples ===')
for f, v in fi_fault.head(10).items():
    print(f'  {f}: {v:.4f}')

print('\n=== Top10 SHAP features: NORMAL samples ===')
for f, v in fi_norm.head(10).items():
    print(f'  {f}: {v:.4f}')

# ---------- Figures ----------
plt.rcParams.update({'font.size': 9})

# Fig 1: global SHAP bar
fig, ax = plt.subplots(figsize=(8, 6))
fi_shap.head(15).plot(kind='barh', ax=ax, color='steelblue')
ax.set_title('Global Mean |SHAP| — Top 15 Features (Real Data, Test Set)')
ax.invert_yaxis()
ax.set_xlabel('mean |SHAP value|')
plt.tight_layout()
fig.savefig(f'{OUT}/routeA_shap_global.png', dpi=150)
plt.close(fig)

# Fig 2: fault vs normal top features
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fi_fault.head(12).plot(kind='barh', ax=axes[0], color='crimson')
axes[0].set_title('Fault Samples — Top 12 Features')
axes[0].invert_yaxis()
fi_norm.head(12).plot(kind='barh', ax=axes[1], color='steelblue')
axes[1].set_title('Normal Samples — Top 12 Features')
axes[1].invert_yaxis()
plt.tight_layout()
fig.savefig(f'{OUT}/routeA_shap_fault_vs_normal.png', dpi=150)
plt.close(fig)

# Fig 3: SHAP summary (beeswarm) — sample 500 test points
np.random.seed(42)
sample_idx = np.random.choice(len(X_test), min(500, len(X_test)), replace=False)
fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values_test[sample_idx], X_test.iloc[sample_idx],
                  max_display=15, show=False)
plt.title('SHAP Summary (500 test samples)')
plt.tight_layout()
fig.savefig(f'{OUT}/routeA_shap_summary.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('\nSaved figures: routeA_shap_global.png, routeA_shap_fault_vs_normal.png, routeA_shap_summary.png')

# ---------- Save ----------
fi_shap.to_frame('mean_abs_shap').to_csv(f'{OUT}/routeA_shap_feature_importance.csv')
results = {
    'model': 'XGBoost',
    'test_metrics': {
        'acc': float(accuracy_score(y_test, pred)),
        'prec': float(precision_score(y_test, pred, zero_division=0)),
        'recall': float(recall_score(y_test, pred)),
        'f1': float(f1_score(y_test, pred)),
        'auc': float(roc_auc_score(y_test, prob)),
    },
    'n_fault_test': int(y_test.sum()),
    'top15_shap': fi_shap.head(15).to_dict(),
    'top10_shap_fault': fi_fault.head(10).to_dict(),
    'top10_shap_normal': fi_norm.head(10).to_dict(),
}
json.dump(results, open(f'{OUT}/routeA_shap_results.json', 'w'), indent=2, default=str)
print('\nSaved routeA_shap_results.json')

# Save pred/prob for later
np.save(f'{OUT}/routeA_test_pred.npy', pred)
np.save(f'{OUT}/routeA_test_prob.npy', prob)
np.save(f'{OUT}/routeA_shap_values.npy', shap_values_test)
print('Saved predictions & SHAP values.')
