#!/usr/bin/env python3
"""build_probe_dataset.py — Phase 0 探路数据集构建（研究方案 v1.0 §3 规格）

输入：/Users/arthas/git/excharge/data/real/all_data.parquet（155.6万采样点，31,449 事务）
输出：earlywarning/data/probe_dataset.parquet
  - 事务级聚合：6 通道序列已排序（按 end_time）
  - 前缀切窗 τ ∈ {1,2,3,5,10,15,20,30} 分钟（时间口径）——记录每个事务在前缀 τ 内的"可见行"
  - 双谱系标签：startup(<30min) / run(>=30min) / normal
  - 事务级 owner 信息（owner1-6 训练 / owner7-8 测试由 probe 脚本切分）

时间语义（实测确认）：
  - begin_time: 会话开始时刻，事务内恒定
  - end_time:   采样时刻，逐行 +60s（99.6% 间隔<=70s）→ 行级时间戳
  - 前缀 τ 的可见行 = end_time - begin_time <= τ（含端点）
"""

import pandas as pd
import numpy as np
import os
import sys
import time

SRC = '/Users/arthas/git/excharge/data/real/all_data.parquet'
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
OUT = os.path.join(OUT_DIR, 'probe_dataset.parquet')

CHANNELS = ['chargingv', 'charginga', 'out_power',
            'charging_gun_temperature1', 'charging_gun_temperature2', 'current_soc']
PREFIX_MINUTES = [1, 2, 3, 5, 10, 15, 20, 30]
FAULT_THRESH_MIN = 30  # 启动型 <30min, 运行型 >=30min
MIN_ROWS_PREFIX = 2    # 前缀内至少 2 行才认为有信号（避免单点抖动）

t0 = time.time()
print('[1/4] 读取 all_data.parquet ...', flush=True)
df = pd.read_parquet(SRC)
df['begin_time'] = pd.to_datetime(df['begin_time'])
df['end_time'] = pd.to_datetime(df['end_time'])
print(f'  原始行 {len(df):,} | 事务 {df["transaction_id"].nunique():,}', flush=True)

print('[2/4] 按事务排序并计算会话级变量 ...', flush=True)
df = df.sort_values(['transaction_id', 'end_time']).reset_index(drop=True)
# 会话时长（分钟）：用 end_time.max - begin_time.min 近似（end_time 是采样戳）
gb_begin = df.groupby('transaction_id')['begin_time'].min()
gb_endmax = df.groupby('transaction_id')['end_time'].max()
gb_endmin = df.groupby('transaction_id')['end_time'].min()
gb_label = df.groupby('transaction_id')['label'].max()
gb_owner = df.groupby('transaction_id')['owner'].first()
# class_judge 取出现最多的（事务级机制码）；types 同
gb_judge = df.groupby('transaction_id')['class_judge'].agg(lambda s: s.mode().iloc[0] if len(s) else -1)
gb_types = df.groupby('transaction_id')['types'].agg(lambda s: s.mode().iloc[0] if len(s) else -1)

meta = pd.DataFrame({
    'transaction_id': gb_begin.index,
    'begin_time': gb_begin.values,
    'end_time_min': gb_endmin.values,   # 第一个采样点时刻
    'end_time_max': gb_endmax.values,   # 最后一个采样点时刻
    'dur_min': (gb_endmax - gb_begin).dt.total_seconds().values / 60.0,
    'label': gb_label.values,
    'owner': gb_owner.values,
    'class_judge': gb_judge.values,
    'types': gb_types.values,
}).reset_index(drop=True)

# 双谱系标签
def family_of(r):
    if r['label'] == 0:
        return 'normal'
    return 'startup' if r['dur_min'] < FAULT_THRESH_MIN else 'run'

meta['family'] = meta.apply(family_of, axis=1)
print(f'  事务总数 {len(meta):,}')
print('  谱系分布:', meta['family'].value_counts().to_dict(), flush=True)

print('[3/4] 计算每事务的采样时间戳数组 + 前缀行数 ...', flush=True)
# 将 end_time 转成相对 begin 的分钟偏移，逐事务数组
df['offset_min'] = (df['end_time'] - df['begin_time']).dt.total_seconds() / 60.0
grp_offsets = df.groupby('transaction_id')['offset_min'].apply(lambda x: np.asarray(x, dtype=np.float64))
meta['offsets'] = meta['transaction_id'].map(grp_offsets).values

# 各前缀 τ 内的可见行数 n_visible(τ) = #{offset_min <= τ}
for tau in PREFIX_MINUTES:
    meta[f'n_pre_{tau}'] = meta['offsets'].apply(lambda a: int(np.sum(a <= tau + 1e-9)))
    # 该前缀 τ 内是否达到最小行数（有信号）
    meta[f'has_pre_{tau}'] = (meta[f'n_pre_{tau}'] >= MIN_ROWS_PREFIX).astype(int)

# 会话总行数（= 该事务采样点数）
meta['n_total_rows'] = meta['offsets'].apply(len)
print('  每事务行数: min=%d med=%.0f max=%d' % (meta['n_total_rows'].min(),
      meta['n_total_rows'].median(), meta['n_total_rows'].max()), flush=True)

print('[4/4] 保存探路数据集 ...', flush=True)
os.makedirs(OUT_DIR, exist_ok=True)
meta.to_parquet(OUT)
print(f'  已保存 {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB)', flush=True)

print('\n=== 探路数据集预览 ===', flush=True)
print('列:', [c for c in meta.columns if c != 'offsets'], flush=True)
print('\n按 owner × family 分布:', flush=True)
print(pd.crosstab(meta['owner'], meta['family']).to_string(), flush=True)

# 各前缀可用样本统计（有信号 = has_pre=1）
print('\n各前缀 τ 可用样本（has_pre=1）:', flush=True)
for tau in PREFIX_MINUTES:
    sub = meta[meta[f'has_pre_{tau}'] == 1]
    if len(sub):
        print(f'  τ={tau:2d}min: {len(sub):6,} 事务 | 故障 {int(sub["label"].sum()):5,} '
              f'({sub["label"].mean()*100:5.2f}%) | 启动型 {int((sub["family"]=="startup").sum()):5,} '
              f'| 运行型 {int((sub["family"]=="run").sum()):5,}', flush=True)

print(f'\n总耗时 {time.time()-t0:.0f}s', flush=True)
