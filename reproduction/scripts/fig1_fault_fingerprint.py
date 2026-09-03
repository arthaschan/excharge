#!/usr/bin/env python3
"""图 1：故障指纹 —— 故障 vs 正常样本特征均值对比（原始单位）。

与论文 §4.3「故障指纹」一致：对测试集(owner7-8, 129 故障 / 2647 正常)按
transaction_id 分组，用 all_data.parquet 的原始信号计算特征均值，故障减正常差值
标注在柱上。数值为均值(mean)，与 §4.5 的中位数口径不同。

输出: figures/fig1_fault_fingerprint.png
依赖: data/real/fusion_data.pkl(测试集 tx/y) + data/real/all_data.parquet(原始信号)
"""
import os, pickle, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = _ROOT + '/data/real/'
FIG = _ROOT + '/figures/'
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10,
                     'legend.fontsize': 9, 'figure.dpi': 150})

D = pickle.load(open(f'{DATA}/fusion_data.pkl', 'rb'))
test_tx = D['test']['tx']
yte = np.asarray(D['test']['y'])
df = pd.read_parquet(f'{DATA}/all_data.parquet')
g = df.groupby('transaction_id')

# 非电压特征（可比尺度）+ 电压特征（V 尺度）
feats = ['soc_last', 'soc_delta', 'p_last', 'p_mean', 'duration_min', 'total_kwh', 't1_mean', 't2_mean']
labels = ['End SOC (%)', 'SOC delta (%)', 'End power (kW)', 'Mean power (kW)',
          'Duration (min)', 'Energy (kWh)', 'Gun temp1 (C)', 'Gun temp2 (C)']
feats_v = ['v_min', 'v_mean', 'v_max']
labels_v = ['Min voltage (V)', 'Mean voltage (V)', 'Max voltage (V)']


def profile(tx):
    s = g.get_group(tx).sort_values('begin_time')
    v = s.chargingv.values.astype(float)
    p = s.out_power.values.astype(float)
    soc = s.current_soc.values.astype(float)
    t1 = s.charging_gun_temperature1.values.astype(float)
    t2 = s.charging_gun_temperature2.values.astype(float)
    return dict(soc_last=soc[-1], soc_delta=soc[-1] - soc[0],
                p_last=p[-1], p_mean=p.mean(),
                duration_min=s.total_charging_min.max(), total_kwh=s.total_charging_kwh.max(),
                t1_mean=np.nanmean(t1), t2_mean=np.nanmean(t2),
                v_min=v.min(), v_mean=v.mean(), v_max=v.max())


rows = [profile(tx) for tx in test_tx]
X = pd.DataFrame(rows)
fault = X[yte == 1]
norm = X[yte == 0]
print(f'fault n={len(fault)}, normal n={len(norm)}', flush=True)

f_means = [fault[f].mean() for f in feats]
n_means = [norm[f].mean() for f in feats]
f_means_v = [fault[f].mean() for f in feats_v]
n_means_v = [norm[f].mean() for f in feats_v]
# 打印核对（对应论文 §4.3 故障指纹表）
print('=== 非电压特征均值 (故障 / 正常 / 差) ===', flush=True)
for f, fm, nm in zip(feats, f_means, n_means):
    print(f'  {f:14s} {fm:.2f} / {nm:.2f} / {fm-nm:+.2f}', flush=True)
print('=== 电压特征均值 ===', flush=True)
for f, fm, nm in zip(feats_v, f_means_v, n_means_v):
    print(f'  {f:14s} {fm:.2f} / {nm:.2f} / {fm-nm:+.2f}', flush=True)

fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [2.6, 1]})
x = np.arange(len(feats)); w = 0.36
axes[0].bar(x - w / 2, n_means, w, label='Normal (n=2,647)', color='#5b8db8', edgecolor='white')
axes[0].bar(x + w / 2, f_means, w, label='Fault (n=129)', color='#d1495b', edgecolor='white')
for i, (fn, ff) in enumerate(zip(n_means, f_means)):
    axes[0].annotate(f'{ff-fn:+.1f}', xy=(i + w / 2, ff), ha='center', va='bottom',
                     fontsize=8, fontweight='bold', color='#d1495b')
axes[0].set_xticks(x); axes[0].set_xticklabels(labels, rotation=20, ha='right')
axes[0].set_ylabel('Feature mean'); axes[0].set_title('(a) Non-voltage features')
axes[0].legend(loc='upper right', fontsize=8)
axes[0].spines['top'].set_visible(False); axes[0].spines['right'].set_visible(False)

xv = np.arange(len(feats_v))
axes[1].bar(xv - w / 2, n_means_v, w, color='#5b8db8', edgecolor='white')
axes[1].bar(xv + w / 2, f_means_v, w, color='#d1495b', edgecolor='white')
for i, (fn, ff) in enumerate(zip(n_means_v, f_means_v)):
    axes[1].annotate(f'{ff-fn:+.1f}', xy=(i + w / 2, ff), ha='center', va='bottom',
                     fontsize=8, fontweight='bold', color='#d1495b')
axes[1].set_xticks(xv); axes[1].set_xticklabels(labels_v, rotation=20, ha='right')
axes[1].set_ylabel('Feature mean (V)'); axes[1].set_title('(b) Voltage features')
axes[1].spines['top'].set_visible(False); axes[1].spines['right'].set_visible(False)

fig.suptitle('Fault Fingerprint: Feature Means — Fault vs Normal\n(Real charging data, new stations)',
             fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(f'{FIG}/fig1_fault_fingerprint.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print('Saved figures/fig1_fault_fingerprint.png', flush=True)
