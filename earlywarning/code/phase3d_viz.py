#!/usr/bin/env python3
"""phase3d_viz.py — Phase 3d: 样例会话可视化 (论文素材)
3 样例: startup 成功预警 / run 成功预警 / normal 对照
每样例: 6 通道时序(前 25min) + EAR 竖线(数据驱动,仅故障) + 绿色可见前缀区
输出: docs/fig_ear_case_{startup,run,normal}.png

修复记录 (2026-09-04): CSV 32 位 transaction_id 被 pandas 解析为 object(Python int),
与 str 目标 == 恒 False → read_csv 需 dtype={'transaction_id': str}。
同时 EAR/lead 改为从 phase3_ear_by_txn.csv 数据驱动读取,normal 对照不画误导性 EAR 竖线。
"""
import pandas as pd, numpy as np, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, 'data'); OUT = os.path.join(BASE, 'docs')
CH6 = ['chargingv(V)', 'charginga(A)', 'out_power(kW)',
       'gunT1(℃)', 'gunT2(℃)', 'SOC(%)']
# 每通道独立配色(论文图可读性)
PAL = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#17becf']
XMAX = 25.0  # 展示窗口: 前 25 min

d = pd.read_parquet(os.path.join(DATA, 'prefix_dataset_full.parquet'))
# 修复: 32 位 transaction_id 必须按 str 读,否则 object(int) 与 str 目标无法匹配
res = pd.read_csv(os.path.join(OUT, 'phase3_ear_by_txn.csv'), dtype={'transaction_id': str})
res = res.set_index('transaction_id')

CASES = [
    ('startup', '44020300500000292209270600964110'),
    ('run',     '44020300500000282403060000085510'),
    ('normal',  '44020100400000032302220202570510'),
]

def load_case(tid):
    r = d[d['transaction_id'] == tid].iloc[0]
    vals = np.asarray(r['vals_flat'], np.float32).reshape(-1, 6)
    offs = np.asarray(r['offsets'], np.float64)
    return vals, offs, float(r['dur_min']), r['family']

for fam, tid in CASES:
    vals, offs, dur, fam_true = load_case(tid)
    # EAR / lead 数据驱动(仅故障族在 CSV 预警人群内; normal 无)
    has_ear = (tid in res.index)
    ear = float(res.loc[tid]['ear']) if has_ear else None
    lead = float(res.loc[tid]['lead_min']) if has_ear else None
    # 展示窗口截断
    m = offs <= XMAX
    offs_v, vals_v = offs[m], vals[m]

    fig, axes = plt.subplots(6, 1, figsize=(10, 12), sharex=True)
    for ch in range(6):
        ax = axes[ch]
        ax.plot(offs_v, vals_v[:, ch], lw=1.4, color=PAL[ch])
        if has_ear:  # 仅故障样例: 红色 EAR 竖线 + 绿色可见前缀区
            ax.axvline(ear, color='red', ls='--', lw=1.5, alpha=0.9)
            ax.axvspan(0, ear, color='green', alpha=0.06)
        ax.set_ylabel(CH6[ch], fontsize=10)
        ax.set_xlim(0, XMAX)
        ax.grid(alpha=0.3)
    if has_ear:
        axes[0].set_title(
            f'{fam} fault case | dur={dur:.1f}min | EAR={ear:.0f}min (red line) | '
            f'lead={lead:.1f}min | green shade=model-visible prefix', fontsize=12)
    else:
        axes[0].set_title(f'normal case | dur={dur:.1f}min | no alarm', fontsize=12)
    axes[-1].set_xlabel('time since plug-in (min)', fontsize=10)
    fig.tight_layout()
    fp = os.path.join(OUT, f'fig_ear_case_{fam}.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    if has_ear:
        print(f'{fam:>7} {tid[-6:]} dur={dur:.1f} EAR={ear:.0f} lead={lead:.1f} → {fp}')
    else:
        print(f'{fam:>7} {tid[-6:]} dur={dur:.1f} (normal 对照, 非 EAR 人群) → {fp}')
print('done')
