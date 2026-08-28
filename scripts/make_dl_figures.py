#!/usr/bin/env python3
"""生成深度模型版论文配图：
1. fig_sensitivity.png: 敏感性分析(逐通道置零 vs 基线, 双指标 Recall/AUC)
2. fig_roc_pr.png: Bi-LSTM 跨站点 ROC + PR 曲线
全英文标注, 无中文字形问题。
"""
import numpy as np, json, pickle, warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

DOCS = '/Users/arthas/git/excharge/docs/'
FIG = '/Users/arthas/git/excharge/paper/figures/'
import os; os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12, 'font.family': 'DejaVu Sans'})

# ============ 图1: 敏感性分析 ============
sens = json.load(open(f'{DOCS}/routeC_bilstm_sensitivity.json'))
base = sens['base']
chs = sens['channel_sensitivity']
labels = [chs[k]['en'] for k in sens['feats']]
recall_drop = [chs[k]['Recall_drop'] for k in sens['feats']]
auc_drop = [chs[k]['AUC_drop'] for k in sens['feats']]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

# 左: Recall 变化
ax = axes[0]
colors = ['#d62728' if x > 0 else '#2ca02c' for x in recall_drop]
bars = ax.barh(range(len(labels)), recall_drop, color=colors, alpha=0.85)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels)
ax.axvline(0, color='k', linewidth=0.8)
ax.set_xlabel('Δ Recall (zero-out − baseline)')
ax.set_title('(a) Channel Sensitivity: Recall Drop')
ax.invert_yaxis()
for i, v in enumerate(recall_drop):
    ax.text(v + (0.01 if v >= 0 else -0.01), i, f'{v:+.2%}', va='center',
            ha='left' if v >= 0 else 'right', fontsize=9)

# 右: AUC 变化
ax = axes[1]
colors = ['#d62728' if x > 0 else '#2ca02c' for x in auc_drop]
bars = ax.barh(range(len(labels)), auc_drop, color=colors, alpha=0.85)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels)
ax.axvline(0, color='k', linewidth=0.8)
ax.set_xlabel('Δ AUC (zero-out − baseline)')
ax.set_title('(b) Channel Sensitivity: AUC Drop')
ax.invert_yaxis()
for i, v in enumerate(auc_drop):
    ax.text(v + (0.005 if v >= 0 else -0.005), i, f'{v:+.3f}', va='center',
            ha='left' if v >= 0 else 'right', fontsize=9)

plt.tight_layout()
plt.savefig(f'{FIG}/fig_sensitivity.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig_sensitivity.png', flush=True)

# ============ 图2: Bi-LSTM ROC + PR ============
with open('/Users/arthas/git/excharge/data/real/seq_tensors.pkl', 'rb') as f:
    d = pickle.load(f)
y_te = d['y_te']
prob = np.load(f'{DOCS}/routeC_bilstm_prob.npy')

fpr, tpr, _ = roc_curve(y_te, prob)
roc_auc = auc(fpr, tpr)
precision, recall, _ = precision_recall_curve(y_te, prob)
pr_auc = average_precision_score(y_te, prob)
baseline = y_te.mean()

fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

ax = axes[0]
ax.plot(fpr, tpr, color='#1f77b4', lw=2, label=f'Bi-LSTM (AUC = {roc_auc:.3f})')
ax.plot([0, 1], [0, 1], color='gray', lw=1, ls='--', label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('(a) ROC Curve (cross-site, owner 7-8)')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(recall, precision, color='#ff7f0e', lw=2, label=f'Bi-LSTM (PR-AUC = {pr_auc:.3f})')
ax.axhline(baseline, color='gray', lw=1, ls='--', label=f'Random baseline ({baseline:.3f})')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('(b) PR Curve (cross-site, owner 7-8)')
ax.legend(loc='upper right')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{FIG}/fig_roc_pr.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved fig_roc_pr.png', flush=True)
print(f'ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}, baseline={baseline:.4f}', flush=True)
