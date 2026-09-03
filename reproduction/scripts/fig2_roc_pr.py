#!/usr/bin/env python3
"""图 2：跨站点测试集 ROC 曲线(左) + PR 曲线(右)，四模型对比。

四模型概率均来自前置步骤保存的文件（按 run_all.sh 顺序执行后即存在）：
  Token-Attn 7-seed 集成: docs/p0a_tokenattn_ensemble_prob.npy
  LightGBM               : docs/gbdt_lightgbm_prob.npy
  XGBoost(62 维)         : docs/gbdt_xgboost62_prob.npy
  Bi-LSTM(端到端)         : docs/routeC_bilstm_prob.npy
标签: data/real/fusion_data.pkl 的 test.y。

输出: figures/fig2_roc_pr.png
"""
import os, pickle, warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = _ROOT + '/data/real/'
DOCS = _ROOT + '/docs/'
FIG = _ROOT + '/figures/'
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
                     'legend.fontsize': 9, 'figure.dpi': 150})

D = pickle.load(open(f'{DATA}/fusion_data.pkl', 'rb'))
yte = np.asarray(D['test']['y'])

models = {
    'Token-Attn (7-seed ens)': ('p0a_tokenattn_ensemble_prob.npy', '#d1495b'),
    'LightGBM': ('gbdt_lightgbm_prob.npy', '#5b8db8'),
    'XGBoost': ('gbdt_xgboost62_prob.npy', '#3a7d44'),
    'Bi-LSTM (end-to-end)': ('routeC_bilstm_prob.npy', '#e8923a'),
}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for name, (fn, color) in models.items():
    prob = np.load(f'{DOCS}/{fn}')
    fpr, tpr, _ = roc_curve(yte, prob)
    a = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(yte, prob)
    ap = average_precision_score(yte, prob)
    axes[0].plot(fpr, tpr, lw=2, color=color, label=f'{name} (AUC={a:.3f})')
    axes[1].plot(rec, prec, lw=2, color=color, label=f'{name} (PR-AUC={ap:.3f})')

axes[0].plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
axes[1].axhline(yte.mean(), color='gray', ls='--', lw=1, alpha=0.7,
                label=f'Random (PR-AUC={yte.mean():.3f})')
axes[0].set_xlabel('False Positive Rate'); axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC — Cross-Station Test (owner 7-8)')
axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
axes[1].set_title('PR — Cross-Station Test (owner 7-8)')
for ax in axes:
    ax.legend(loc='lower right', fontsize=8)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
fig.savefig(f'{FIG}/fig2_roc_pr.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print('Saved figures/fig2_roc_pr.png', flush=True)
