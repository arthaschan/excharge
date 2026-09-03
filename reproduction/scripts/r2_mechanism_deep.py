#!/usr/bin/env python3
"""机制分析深化 (回应审稿意见 #4 温度矛盾 + #8 置换重要性量级)。

#4 温度矛盾: 故障定义含"枪温过高断开"(类型2), 但故障样本枪温均值反而更低。
   → 需证明: 类型2(高温断开)是少数; 温度特征居首是因为它编码"充电完整性/时长", 而非"过热"本身。
#8 置换重要性量级: t2_max ΔPR-AUC=0.488 接近"移除32维"的0.61, Top5全温度, 可疑。
   → 需证明: 温度特征高度冗余于"时长"(故障提前终止→时长短→温度低), 置换温度≈破坏时长信号。
输出: docs/r2_mechanism_deep.json
"""
import os, pickle, warnings, json
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = _ROOT + '/data/real/'
OUT = _ROOT + '/docs/'

D = pickle.load(open(f'{DATA}/fusion_data.pkl', 'rb'))
df = pd.read_parquet(f'{DATA}/all_data.parquet')
g = df.groupby('transaction_id')

test_tx = D['test']['tx']; yte = np.asarray(D['test']['y'])
fault_tx = [tx for tx, y in zip(test_tx, yte) if y == 1]
norm_tx = [tx for tx, y in zip(test_tx, yte) if y == 0]

def prof(tx):
    s = g.get_group(tx).sort_values('begin_time')
    t1 = s.charging_gun_temperature1.values.astype(float)
    t2 = s.charging_gun_temperature2.values.astype(float)
    soc = s.current_soc.values.astype(float)
    return dict(tx=tx, t1_max=t1.max(), t2_max=t2.max(), t1_mean=t1.mean(),
                t2_mean=t2.mean(), t2_last=t2[-1], t1_last=t1[-1],
                soc_end=soc[-1], soc_delta=soc[-1]-soc[0], dur=s.total_charging_min.max())

F = pd.DataFrame([prof(tx) for tx in fault_tx])
N = pd.DataFrame([prof(tx) for tx in norm_tx])
ALL = pd.concat([F.assign(fault=1), N.assign(fault=0)], ignore_index=True)

print(f'故障 {len(F)} / 正常 {len(N)}', flush=True)

# ---- #8: 温度特征与"时长/充电完整性"的相关性 ----
print('\n=== 温度特征 vs 时长/SOC增量 相关 (全部 test 样本) ===', flush=True)
temp_cols = ['t1_max', 't2_max', 't1_mean', 't2_mean', 't2_last', 't1_last']
for c in temp_cols:
    r_dur = float(np.corrcoef(ALL[c], ALL.dur)[0, 1])
    r_soc = float(np.corrcoef(ALL[c], ALL.soc_delta)[0, 1])
    print(f'  {c:8s}: corr(时长)={r_dur:+.3f}  corr(SOC增量)={r_soc:+.3f}', flush=True)

# ---- #8: 置换温度特征 ≈ 破坏时长信号 (温度是时长的高冗余代理) ----
# 关键: 温度特征的"充电完整性"编码 = 故障(短时长) vs 正常(长时长)
print('\n=== 时长 vs 温度 (故障短→温度低) ===', flush=True)
print(f'  故障: 时长中位 {F.dur.median():.0f}min, t2_max中位 {F.t2_max.median():.0f}°C', flush=True)
print(f'  正常: 时长中位 {N.dur.median():.0f}min, t2_max中位 {N.t2_max.median():.0f}°C', flush=True)

# ---- #4: 分型映射到 类型1/类型2 ----
F['type'] = '电气中断型(类型1: 工程上报)'
F.loc[F.t1_max > 55, 'type'] = '超温型(类型2: 枪温过高断开)'
F.loc[F.t2_max > 55, 'type'] = '超温型(类型2: 枪温过高断开)'
tc = F['type'].value_counts()
print('\n=== 分型 → 故障类型映射 ===', flush=True)
for t, c in tc.items():
    print(f'  {t}: {c} ({c/len(F)*100:.1f}%)', flush=True)

# 类型2(高温) 是否真的"温度高"?
hi = F['type'].str.startswith('超温型')
print(f'\n  类型2(超温型) t2_max中位={F.loc[hi,"t2_max"].median():.0f}°C, 时长中位={F.loc[hi,"dur"].median():.0f}min', flush=True)
print(f'  类型1(电气中断型) t2_max中位={F.loc[~hi,"t2_max"].median():.0f}°C, 时长中位={F.loc[~hi,"dur"].median():.0f}min', flush=True)

out = {
    'temperature_duration_corr': {c: {'dur': float(np.corrcoef(ALL[c], ALL.dur)[0,1]),
                                      'soc_delta': float(np.corrcoef(ALL[c], ALL.soc_delta)[0,1])}
                                  for c in temp_cols},
    'fault_normal': {'fault_dur_med': float(F.dur.median()), 'normal_dur_med': float(N.dur.median()),
                     'fault_t2max_med': float(F.t2_max.median()), 'normal_t2max_med': float(N.t2_max.median())},
    'type_mapping': {k: int(v) for k, v in tc.items()},
    'type2_stats': {'t2max_med': float(F.loc[hi, 't2_max'].median()), 'dur_med': float(F.loc[hi, 'dur'].median()),
                    'count': int(hi.sum())},
    'type1_stats': {'t2max_med': float(F.loc[~hi, 't2_max'].median()), 'dur_med': float(F.loc[~hi, 'dur'].median()),
                    'count': int((~hi).sum())},
}
json.dump(out, open(f'{OUT}/r2_mechanism_deep.json', 'w'), indent=2, ensure_ascii=False)
print('\nDONE', flush=True)
