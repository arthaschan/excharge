#!/usr/bin/env python3
"""Generate publication-quality figures for the paper (English labels).
Fig 1: Fault fingerprint comparison (fault vs normal feature means)
Fig 2: ROC curves (XGBoost v1+sel_v2, MLP, RF, Transformer from Route B)
Fig 3: SHAP summary beeswarm (top 15)
Fig 4: SHAP global importance bar (fault vs normal)
"""
import pandas as pd, numpy as np, json, warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import shap

DATA = '/Users/arthas/git/excharge/data/real/'
OUT  = '/Users/arthas/git/excharge/paper/figures/'
import os
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})

# ============ Load data ============
X_test = pd.read_parquet(f'{DATA}/seq_X_test.parquet')
y_test = pd.read_parquet(f'{DATA}/seq_y_test.parquet')['label'].values
X_te57 = pd.read_parquet(f'{DATA}/seq_X_test_v2.parquet')  # v2 for prob alignment
shap_vals = np.load('/Users/arthas/git/excharge/docs/routeA_v2_shap.npy')  # 62-feat model SHAP

# ============ Fig 1: Fault fingerprint ============
fault = X_test[y_test == 1]
norm  = X_test[y_test == 0]
# Two panels: non-voltage features (comparable scale) + voltage (V scale)
feats = ['soc_last', 'soc_delta', 'p_last', 'p_mean', 'duration_min', 'total_kwh', 't1_mean', 't2_mean']
labels = ['End SOC (%)', 'SOC delta (%)', 'End power (kW)', 'Mean power (kW)',
          'Duration (min)', 'Energy (kWh)', 'Gun temp1 (C)', 'Gun temp2 (C)']
feats_v = ['v_min', 'v_mean', 'v_max']
labels_v = ['Min voltage (V)', 'Mean voltage (V)', 'Max voltage (V)']

fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [2.6, 1]})
f_means = [fault[f].mean() for f in feats]
n_means = [norm[f].mean() for f in feats]
x = np.arange(len(feats)); w = 0.36
axes[0].bar(x - w/2, n_means, w, label='Normal (n=2,647)', color='#5b8db8', edgecolor='white')
axes[0].bar(x + w/2, f_means, w, label='Fault (n=129)', color='#d1495b', edgecolor='white')
for i, (fn, ff) in enumerate(zip(n_means, f_means)):
    axes[0].annotate(f'{ff-fn:+.1f}', xy=(i + w/2, ff), ha='center', va='bottom', fontsize=8,
                fontweight='bold', color='#d1495b')
axes[0].set_xticks(x)
axes[0].set_xticklabels(labels, rotation=20, ha='right')
axes[0].set_ylabel('Feature mean')
axes[0].set_title('(a) Non-voltage features')
axes[0].legend(loc='upper right', fontsize=8)
axes[0].spines['top'].set_visible(False); axes[0].spines['right'].set_visible(False)

f_means_v = [fault[f].mean() for f in feats_v]
n_means_v = [norm[f].mean() for f in feats_v]
xv = np.arange(len(feats_v))
axes[1].bar(xv - w/2, n_means_v, w, color='#5b8db8', edgecolor='white')
axes[1].bar(xv + w/2, f_means_v, w, color='#d1495b', edgecolor='white')
for i, (fn, ff) in enumerate(zip(n_means_v, f_means_v)):
    axes[1].annotate(f'{ff-fn:+.1f}', xy=(i + w/2, ff), ha='center', va='bottom', fontsize=8,
                fontweight='bold', color='#d1495b')
axes[1].set_xticks(xv)
axes[1].set_xticklabels(labels_v, rotation=20, ha='right')
axes[1].set_ylabel('Feature mean (V)')
axes[1].set_title('(b) Voltage features')
axes[1].spines['top'].set_visible(False); axes[1].spines['right'].set_visible(False)

fig.suptitle('Fault Fingerprint: Feature Means — Fault vs Normal\n(Real charging data, new stations)', fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(f'{OUT}/fig1_fault_fingerprint.png', dpi=300)
plt.close(fig)
print('Fig1 saved')

# ============ Fig 2: ROC curves ============
# Need probs for all models on same test set. XGBoost v1+sel_v2 saved; others re-run quickly.
import sys
sys.path.insert(0, '/Users/arthas/git/excharge/scripts')
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

X_tr = pd.read_parquet(f'{DATA}/seq_X_train.parquet')
y_tr  = pd.read_parquet(f'{DATA}/seq_y_train.parquet')['label'].values
X2_tr = pd.read_parquet(f'{DATA}/seq_X_train_v2.parquet')
common_tr = X_tr.index.intersection(X2_tr.index)
sel_v2 = ['v_first_third_min','v_sag_from_mean','t1_slope','t2_slope','t1_max_jump','t2_max_jump',
          't1_std_2nd','t2_std_2nd','a_last_third_max','a_first_last_ratio','p_max_jump','soc_rate',
          'soc_last_rate','a_seg_change_2to3','v_last3_vs_first3','a_last3_vs_first3',
          't1_over_40','t2_over_40','t1_over_45','t2_over_45']
sel_v2 = [c for c in sel_v2 if c in X2_tr.columns]
X_tr57 = pd.concat([X_tr.reset_index(drop=True), X2_tr.loc[common_tr, sel_v2].reset_index(drop=True)], axis=1)
X_te57 = pd.concat([X_test.reset_index(drop=True), X_te57.loc[X_test.index.intersection(X_te57.index), sel_v2].reset_index(drop=True)], axis=1)
y_tr57 = y_tr[X_tr.index.get_indexer(common_tr)]

spw = (y_tr57 == 0).sum() / (y_tr57 == 1).sum()

models = {
    'XGBoost (57 feat)': XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                        scale_pos_weight=spw, random_state=42, n_jobs=-1),
    'RandomForest': RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1,
                                            class_weight='balanced'),
    'MLP': MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42),
}
probs = {}
for name, mdl in models.items():
    if name == 'XGBoost (57 feat)':
        mdl.fit(X_tr57, y_tr57)
        probs[name] = mdl.predict_proba(X_te57)[:, 1]
    else:
        mdl.fit(X_tr, y_tr)
        probs[name] = mdl.predict_proba(X_test)[:, 1]

# Load transformer probs (Route B)
try:
    tr_prob = np.load('/Users/arthas/git/excharge/docs/routeB_transformer_prob.npy')
    # align: Route B test had 7,768 windows, our seq test has 2,776 — use per-seq aggregation instead
    # fallback: use saved Route B json metrics only
    probs['Transformer'] = None
except Exception as e:
    probs['Transformer'] = None

fig, ax = plt.subplots(figsize=(7, 6))
colors = {'XGBoost (57 feat)': '#d1495b', 'RandomForest': '#5b8db8', 'MLP': '#3a7d44'}
for name, p in probs.items():
    if p is None:
        continue
    fpr, tpr, _ = roc_curve(y_test, p)
    a = auc(fpr, tpr)
    ax.plot(fpr, tpr, lw=2, color=colors[name], label=f'{name} (AUC={a:.3f})')
# Nature baseline point (from paper: Recall=73.56%, Acc=92.43%) — approximate FPR from Acc
ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves — Cross-Station Test (owner 7-8)')
ax.legend(loc='lower right')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
fig.savefig(f'{OUT}/fig2_roc_curves.png', dpi=300)
plt.close(fig)
print('Fig2 saved')

# ============ Fig 3: SHAP summary ============
explainer_path = '/Users/arthas/git/excharge/docs/routeA_v2_shap.npy'
fig, ax = plt.subplots(figsize=(9, 7))
np.random.seed(42)
# v2 model used 62 features; load the v2 test features for SHAP plot
X_te_v2_full = pd.read_parquet(f'{DATA}/seq_X_test_v2.parquet')
shap_v2 = np.load(explainer_path)
sample_idx = np.random.choice(len(X_te_v2_full), min(400, len(X_te_v2_full)), replace=False)
shap.summary_plot(shap_v2[sample_idx], X_te_v2_full.iloc[sample_idx], max_display=15, show=False)
plt.title('SHAP Summary — Top 15 Features (new stations)')
plt.tight_layout()
fig.savefig(f'{OUT}/fig3_shap_summary.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print('Fig3 saved')

# ============ Fig 4: SHAP global bar fault vs normal ============
fi_all = pd.Series(np.abs(shap_v2).mean(axis=0), index=X_te_v2_full.columns).sort_values(ascending=False)
fault_idx = np.where(y_test == 1)[0]
fi_fault = pd.Series(np.abs(shap_v2[fault_idx]).mean(axis=0), index=X_te_v2_full.columns).sort_values(ascending=False)
top = fi_fault.head(12).index.tolist()

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
axes[0].barh(top[::-1], fi_fault[top[::-1]].values, color='#d1495b')
axes[0].set_title('Fault samples (n=129)')
axes[0].invert_yaxis() if False else None
axes[1].barh(top[::-1], fi_all[top[::-1]].values, color='#5b8db8')
axes[1].set_title('All test samples (n=2,776)')
for ax_ in axes:
    ax_.set_xlabel('mean |SHAP|')
    ax_.spines['top'].set_visible(False); ax_.spines['right'].set_visible(False)
plt.tight_layout()
fig.savefig(f'{OUT}/fig4_shap_fault_vs_all.png', dpi=300)
plt.close(fig)
print('Fig4 saved')
print('\nAll figures saved to paper/figures/')
