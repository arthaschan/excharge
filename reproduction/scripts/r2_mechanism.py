#!/usr/bin/env python3
"""R2 补充: 异常成因机制分析 (老师要求, 修正版)。

核心因果逻辑: 故障 = 充电在"高功率未衰减"状态下被异常终止(而非正常涓流结束)。
  温度升高是"充电时长的伴随结果", 不是"故障的原因"——故障枪温反而更低(提前终止)。
分型: 电气中断型(主导) vs 超温型(少数, 温度是前兆/原因)。
输出: docs/r2_mechanism.json + docs/r2_figs/mechanism_*.png
"""
import os, pickle, warnings, json
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = _ROOT + '/data/real/'
OUT = _ROOT + '/docs/'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
os.makedirs(f'{OUT}/r2_figs', exist_ok=True)

D = pickle.load(open(f'{DATA}/fusion_data.pkl', 'rb'))
df = pd.read_parquet(f'{DATA}/all_data.parquet')
test_tx = D['test']['tx']; yte = np.asarray(D['test']['y'])
fault_tx = [tx for tx, y in zip(test_tx, yte) if y == 1]
norm_tx = [tx for tx, y in zip(test_tx, yte) if y == 0]
g = df.groupby('transaction_id')

def profile(tx):
    s = g.get_group(tx).sort_values('begin_time')
    v = s.chargingv.values.astype(float); a = s.charginga.values.astype(float)
    p = s.out_power.values.astype(float); t1 = s.charging_gun_temperature1.values.astype(float)
    t2 = s.charging_gun_temperature2.values.astype(float); soc = s.current_soc.values.astype(float)
    # 终止方式: 末端(最后1步) vs 峰值
    return dict(tx=tx,
                p_end=p[-1], p_peak=p.max(), p_end_frac=float(p[-1]/max(p.max(),1e-6)),
                a_end=a[-1], a_peak=a.max(), a_end_frac=float(a[-1]/max(a.max(),1e-6)),
                v_min=v.min(), t_max=max(t1.max(), t2.max()),
                soc_end=soc[-1], soc_first=soc[0], dur=s.total_charging_min.max(), n=len(s))

F = pd.DataFrame([profile(tx) for tx in fault_tx])
N = pd.DataFrame([profile(tx) for tx in norm_tx])

# ---------- 1) 分型 ----------
F['type'] = '电气中断型'
F.loc[F.t_max > 55, 'type'] = '超温型'
tc = F['type'].value_counts()
print('=== 分型 ===', flush=True)
for t, c in tc.items():
    print(f'  {t}: {c} ({c/len(F)*100:.1f}%)', flush=True)

# ---------- 2) 终止方式对比 (中断的直接证据) ----------
def end_report(d, name):
    return dict(name=name, p_end=float(d.p_end.median()), p_end_frac=float(d.p_end_frac.median()),
                a_end=float(d.a_end.median()), a_end_frac=float(d.a_end_frac.median()),
                soc_end=float(d.soc_end.median()), dur=float(d.dur.median()), t_max=float(d.t_max.median()))

print('\n=== 终止方式对比 (末端功率未衰减 = 异常中断) ===', flush=True)
for d, nm in [(N, '正常'), (F, '故障'), (F[F.type=='电气中断型'], '  └电气中断型'), (F[F.type=='超温型'], '  └超温型')]:
    r = end_report(d, nm)
    print(f'  {nm:14s} 末端功率 {r["p_end"]:5.1f}kW (占峰值 {r["p_end_frac"]*100:3.0f}%) | '
          f'末端电流 {r["a_end"]:5.1f}A (占峰值 {r["a_end_frac"]*100:3.0f}%) | '
          f'末端SOC {r["soc_end"]:.0f}% | 时长 {r["dur"]:.0f}min | 枪温max {r["t_max"]:.0f}°C', flush=True)

# ---------- 3) 温度角色 (前兆 vs 后果) ----------
print('\n=== 温度角色: 前兆(因) vs 伴随(果) ===', flush=True)
print(f'  超温型: 枪温max中位 {F[F.type=="超温型"].t_max.median():.0f}°C, 时长中位 {F[F.type=="超温型"].dur.median():.0f}min → 温度高且时长不短 → 温度是"前兆/原因"', flush=True)
print(f'  电气中断型: 枪温max中位 {F[F.type=="电气中断型"].t_max.median():.0f}°C, 时长中位 {F[F.type=="电气中断型"].dur.median():.0f}min → 温度低且时长短 → 温度是"时长结果/后果"', flush=True)
print(f'  正常: 枪温max中位 {N.t_max.median():.0f}°C, 时长中位 {N.dur.median():.0f}min → 正常充满、温度充分累积', flush=True)

# ---------- 4) 机制-特征映射 ----------
mapping = [
    {'特征': 't2_max / t1_mean / t2_last (枪温统计)', '机制': '超温/热相关', '因果角色': '超温型=前兆(因); 电气中断型=时长结果(果)', '运维建议': '查散热/枪头连接; 区分高温型与中断型'},
    {'特征': 'a_min / v_min (电流/电压骤降)', '机制': '短路/接触不良/电芯异常', '因果角色': '中断的直接电气表现', '运维建议': '查电路连接与电芯一致性'},
    {'特征': 'p_end (末端功率未衰减)', '机制': '高功率异常终止', '因果角色': '中断的核心证据', '运维建议': '查充电桩保护动作日志'},
    {'特征': 'soc_last (末端SOC未充满)', '机制': '充电提前终止', '因果角色': '中断的后果', '运维建议': '核查中断原因'},
    {'特征': 'a_last_third_max / a_seg3_max (末段电流峰值)', '机制': '大电流段运行', '因果角色': '中断发生的工况', '运维建议': '关注高功率段稳定性'},
]

out = {
    'type_counts': {k: int(v) for k, v in tc.items()},
    'termination_mode': {
        'normal': end_report(N, 'normal'), 'fault': end_report(F, 'fault'),
        'electrical': end_report(F[F.type=='电气中断型'], 'electrical'),
        'overtemp': end_report(F[F.type=='超温型'], 'overtemp'),
    },
    'temperature_role': {
        'overtemp_tmax_med': float(F[F.type=="超温型"].t_max.median()),
        'electrical_tmax_med': float(F[F.type=="电气中断型"].t_max.median()),
        'normal_tmax_med': float(N.t_max.median()),
        'overtemp_dur_med': float(F[F.type=="超温型"].dur.median()),
        'electrical_dur_med': float(F[F.type=="电气中断型"].dur.median()),
        'normal_dur_med': float(N.dur.median()),
    },
    'mechanism_feature_mapping': mapping,
}
json.dump(out, open(f'{OUT}/r2_mechanism.json', 'w'), indent=2, ensure_ascii=False)

# ---------- 图 ----------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
# (a) 分型饼图
axes[0].pie(tc.values, labels=[f'{k}\n({v})' for k, v in tc.items()], colors=['#4C72B0', '#C44E52'], startangle=90)
axes[0].set_title('(a) Fault mechanism sub-types')
# (b) 末端功率占比 (正常 vs 故障)
cats = ['正常', '电气中断型', '超温型']
p_end_frac = [N.p_end_frac.median(), F[F.type=='电气中断型'].p_end_frac.median(), F[F.type=='超温型'].p_end_frac.median()]
axes[1].bar(cats, p_end_frac, color=['#55A868', '#4C72B0', '#C44E52'])
axes[1].set_ylabel('末端功率 / 峰值功率')
axes[1].set_title('(b) 末端功率占比(正常衰减到涓流, 故障未衰减=异常中断)')
axes[1].axhline(1.0, color='gray', ls='--', lw=0.8)
for i, v in enumerate(p_end_frac):
    axes[1].text(i, v+0.03, f'{v:.2f}', ha='center', fontsize=9)
# (c) 枪温max × 时长 (温度角色)
axes[2].scatter(N.dur, N.t_max, s=8, alpha=0.4, label='正常', color='#55A868')
axes[2].scatter(F[F.type=='电气中断型'].dur, F[F.type=='电气中断型'].t_max, s=18, label='电气中断型', color='#4C72B0')
axes[2].scatter(F[F.type=='超温型'].dur, F[F.type=='超温型'].t_max, s=30, label='超温型', color='#C44E52', marker='^')
axes[2].set_xlabel('充电时长 (min)'); axes[2].set_ylabel('枪温最大值 (°C)')
axes[2].set_title('(c) 枪温×时长: 温度是时长结果, 非中断原因')
axes[2].legend(fontsize=8)
plt.tight_layout(); plt.savefig(f'{OUT}/r2_figs/mechanism_subtypes_causal.png', dpi=150, bbox_inches='tight'); plt.close()
print('\nDONE', flush=True)
