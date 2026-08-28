#!/usr/bin/env python3
"""融合模型双轨之特征轨: 对 62 维手工特征做 TreeSHAP 归因。
训练 XGBoost (owner1-6) 于 62 维特征, TreeSHAP 解释测试集 (owner7-8),
与序列分支的 1D-GradCAM 互为补充。
输出: docs/fusion_shap_*.png / fusion_shap_results.json / fusion_shap_feature_importance.csv
"""
import pickle, numpy as np, json, time, warnings, os
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score)

DATA = '/Users/arthas/git/excharge/data/real/'
OUT  = '/Users/arthas/git/excharge/docs/'
os.makedirs(OUT, exist_ok=True)

with open(f'{DATA}/fusion_data.pkl', 'rb') as f:
    D = pickle.load(f)
feat_cols = D['feat_cols']
X_train = D['train']['X_feat']; y_train = D['train']['y']
X_val   = D['val']['X_feat'];   y_val   = D['val']['y']
X_test  = D['test']['X_feat'];  y_test  = D['test']['y']
print(f'Train {len(X_train):,} | Val {len(X_val):,} | Test {len(X_test):,} (fault {y_test.sum()})', flush=True)

t0 = time.time()
xgb = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
                    eval_metric='logloss', random_state=42, n_jobs=-1)
xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
print(f'XGBoost (62-dim features) trained in {time.time()-t0:.1f}s', flush=True)
pred = xgb.predict(X_test); prob = xgb.predict_proba(X_test)[:, 1]
print(f'Test: Acc={accuracy_score(y_test,pred):.4f} Prec={precision_score(y_test,pred,zero_division=0):.4f} '
      f'Recall={recall_score(y_test,pred):.4f} F1={f1_score(y_test,pred):.4f} '
      f'AUC={roc_auc_score(y_test,prob):.4f} PR-AUC={average_precision_score(y_test,prob):.4f}', flush=True)

print('Computing TreeSHAP on test set...', flush=True)
t0 = time.time()
explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_test)
if isinstance(shap_values, list):
    shap_values = shap_values[1]   # 取正类
print(f'SHAP done in {time.time()-t0:.0f}s, shape={shap_values.shape}', flush=True)

mean_abs = np.abs(shap_values).mean(axis=0)
fi = dict(zip(feat_cols, mean_abs))
fi_sorted = sorted(fi.items(), key=lambda kv: -kv[1])
print('\n=== Top15 SHAP features (mean |SHAP|) ===')
for f, v in fi_sorted[:15]:
    print(f'  {f}: {v:.4f}')

fault_idx = np.where(y_test == 1)[0]; norm_idx = np.where(y_test == 0)[0]
fault_shap = np.abs(shap_values[fault_idx]).mean(axis=0)
norm_shap  = np.abs(shap_values[norm_idx]).mean(axis=0)

plt.rcParams.update({'font.size': 9})
# Fig: global SHAP bar (top 15)
fig, ax = plt.subplots(figsize=(8, 6))
top = fi_sorted[:15][::-1]
ax.barh([k for k, _ in top], [v for _, v in top], color='steelblue')
ax.set_title('Fusion: Global Mean |SHAP| — Top 15 Handcrafted Features (Test)')
ax.set_xlabel('mean |SHAP value|')
plt.tight_layout(); fig.savefig(f'{OUT}/fusion_shap_global.png', dpi=150); plt.close(fig)

# Fig: fault vs normal top features
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fs = sorted(zip(feat_cols, fault_shap), key=lambda kv: -kv[1])[:12][::-1]
ns = sorted(zip(feat_cols, norm_shap), key=lambda kv: -kv[1])[:12][::-1]
axes[0].barh([k for k, _ in fs], [v for _, v in fs], color='crimson'); axes[0].set_title('Fault Samples — Top 12')
axes[1].barh([k for k, _ in ns], [v for _, v in ns], color='steelblue'); axes[1].set_title('Normal Samples — Top 12')
plt.tight_layout(); fig.savefig(f'{OUT}/fusion_shap_fault_vs_normal.png', dpi=150); plt.close(fig)

# Fig: SHAP summary beeswarm (sample 500)
np.random.seed(42)
sidx = np.random.choice(len(X_test), min(500, len(X_test)), replace=False)
fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values[sidx], X_test[sidx], feature_names=feat_cols, max_display=15, show=False)
plt.title('Fusion SHAP Summary (500 test samples)')
plt.tight_layout(); fig.savefig(f'{OUT}/fusion_shap_summary.png', dpi=150, bbox_inches='tight'); plt.close(fig)
print('Saved fusion_shap_global.png, fusion_shap_fault_vs_normal.png, fusion_shap_summary.png', flush=True)

import pandas as pd
pd.DataFrame({'feature': feat_cols, 'mean_abs_shap': mean_abs}).sort_values(
    'mean_abs_shap', ascending=False).to_csv(f'{OUT}/fusion_shap_feature_importance.csv', index=False)
results = {'n_features': len(feat_cols), 'top15_shap': dict(fi_sorted[:15]),
           'xgb_test': {'acc': float(accuracy_score(y_test, pred)), 'recall': float(recall_score(y_test, pred)),
                        'auc': float(roc_auc_score(y_test, prob)), 'pr_auc': float(average_precision_score(y_test, prob))}}
json.dump(results, open(f'{OUT}/fusion_shap_results.json', 'w'), indent=2, default=str)
np.save(f'{OUT}/fusion_shap_values.npy', shap_values)
print('Saved fusion_shap_results.json + csv + npy', flush=True)
