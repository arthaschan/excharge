#!/usr/bin/env python3
"""extract_prefix_features.py — Phase 0 前缀统计特征提取（研究方案 §3 / E11 防线）

输入：all_data.parquet（6 通道原始）+ probe_dataset.parquet（事务级元数据/前缀行数）
输出：earlywarning/data/prefix_feats.parquet（长表：每事务 × 每前缀口径 一行）

口径：
- 时间切窗 τ ∈ {1,2,3,5,10,20} min：前缀可见行 = end_time - begin_time <= τ
- 进度切窗 p ∈ {10%,25%,50%}：前缀 = 会话前 p% 的行（按 end_time 排序取前 ceil(n*p) 行）
- 存活人群 D_τ = {dur_min >= 前缀时长}（到该前缀时刻会话还活着，deployment-realistic）
- ⚠️ E11 防泄漏：特征只用前缀内行；严禁引入 dur_min / n_total_rows / 未来通道值。
  n_prefix_rows 本身可见（= 到 τ 为止已有几行），允许。
"""

import pandas as pd
import numpy as np
import os, time, sys

SRC = '/Users/arthas/git/excharge/data/real/all_data.parquet'
META = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'probe_dataset.parquet')
OUT = os.path.join(os.path.dirname(META), 'prefix_feats.parquet')

CHANNELS = ['chargingv', 'charginga', 'out_power',
            'charging_gun_temperature1', 'charging_gun_temperature2', 'current_soc']
TIME_TAUS = [1, 2, 3, 5, 10, 20]
PROG_PCTS = [10, 25, 50]
MIN_ROWS = 2   # 前缀至少 2 行才有统计意义

def feats_from_block(block):  # block: np.ndarray (n, 6), 按时间排序
    """对前缀块算 6 通道统计特征。返回特征 dict。block 至少 2 行。"""
    n = block.shape[0]
    f = {}
    for j, ch in enumerate(CHANNELS):
        col = block[:, j].astype(np.float64)
        col = col[np.isfinite(col)]
        if len(col) == 0:
            base = {f'{ch}_mean': np.nan, f'{ch}_last': np.nan, f'{ch}_first': np.nan,
                    f'{ch}_max': np.nan, f'{ch}_min': np.nan, f'{ch}_std': np.nan,
                    f'{ch}_slope': np.nan, f'{ch}_range': np.nan}
        elif len(col) == 1:
            base = {f'{ch}_mean': col[0], f'{ch}_last': col[0], f'{ch}_first': col[0],
                    f'{ch}_max': col[0], f'{ch}_min': col[0], f'{ch}_std': 0.0,
                    f'{ch}_slope': 0.0, f'{ch}_range': 0.0}
        else:
            slope = (col[-1] - col[0]) / (len(col) - 1)  # 每行平均变化
            base = {f'{ch}_mean': float(col.mean()), f'{ch}_last': float(col[-1]),
                    f'{ch}_first': float(col[0]), f'{ch}_max': float(col.max()),
                    f'{ch}_min': float(col.min()), f'{ch}_std': float(col.std()),
                    f'{ch}_slope': float(slope), f'{ch}_range': float(col.max() - col.min())}
        f.update(base)
    # 跨通道业务特征
    p = block[:, 2].astype(np.float64)
    soc = block[:, 5].astype(np.float64)
    t1 = block[:, 3].astype(np.float64)
    # SOC 增量（末-首，反映真实充入进度）
    f['soc_delta'] = float(soc[-1] - soc[0]) if len(soc) else np.nan
    # 功率活跃度：>1kW 行占比（充电是否真的开始）
    f['power_active_ratio'] = float((p > 1).mean()) if len(p) else np.nan
    # 功率达到峰值的位置（0~1，越靠前说明首段即达峰）
    if len(p) and np.nanmax(p) > 0:
        f['power_peak_pos'] = float(np.nanargmax(p)) / (len(p) - 1)
    else:
        f['power_peak_pos'] = np.nan
    # 枪温爬升（末-首）
    f['gunT1_rise'] = float(t1[-1] - t1[0]) if len(t1) else np.nan
    return f

t0 = time.time()
print('[1/3] 读取原始数据 + 事务元数据 ...', flush=True)
df = pd.read_parquet(SRC)
df['begin_time'] = pd.to_datetime(df['begin_time'])
df['end_time'] = pd.to_datetime(df['end_time'])
meta = pd.read_parquet(META)
# 需要 dur_min 判断存活；注意特征里不放进 dur，只用于存活过滤（构建期过滤，不进模型输入）
dur_map = meta.set_index('transaction_id')['dur_min']
family_map = meta.set_index('transaction_id')['family']
owner_map = meta.set_index('transaction_id')['owner']
label_map = meta.set_index('transaction_id')['label']

df = df.sort_values(['transaction_id', 'end_time']).reset_index(drop=True)
df['offset_min'] = (df['end_time'] - df['begin_time']).dt.total_seconds() / 60.0

print('[2/3] 逐事务提取前缀特征（时间 + 进度双口径）...', flush=True)
rows = []
tids = df['transaction_id'].unique()
for i, tid in enumerate(tids):
    sub = df[df['transaction_id'] == tid]
    # 完整通道块（已按 end_time 排序）
    vals = sub[CHANNELS].values  # (n, 6)
    offs = sub['offset_min'].values
    dur = float(dur_map.loc[tid])
    n_all = len(sub)

    # —— 时间切窗 ——
    for tau in TIME_TAUS:
        if dur < tau:          # 存活人群过滤：τ 时刻会话已结束 → 不构成预测对象
            continue
        mask = offs <= tau + 1e-9
        blk = vals[mask]
        if len(blk) < MIN_ROWS:
            continue
        f = feats_from_block(blk)
        f.update({'transaction_id': tid, 'prefix_type': 'time', 'prefix_val': tau,
                  'n_prefix_rows': len(blk), 'owner': owner_map.loc[tid],
                  'family': family_map.loc[tid], 'label': int(label_map.loc[tid])})
        rows.append(f)

    # —— 进度切窗 ——
    for pp in PROG_PCTS:
        n_pre = max(MIN_ROWS, int(np.ceil(n_all * pp / 100.0)))
        if n_pre > n_all:      # 事务太短
            continue
        # 进度前缀的"等效存活"：会话总长必须覆盖该进度已消耗的时间——由定义保证(n_pre<=n_all)
        blk = vals[:n_pre]
        if len(blk) < MIN_ROWS:
            continue
        f = feats_from_block(blk)
        f.update({'transaction_id': tid, 'prefix_type': 'progress', 'prefix_val': pp,
                  'n_prefix_rows': len(blk), 'owner': owner_map.loc[tid],
                  'family': family_map.loc[tid], 'label': int(label_map.loc[tid])})
        rows.append(f)

    if (i + 1) % 5000 == 0:
        print(f'  已处理 {i+1}/{len(tids)} 事务 ({time.time()-t0:.0f}s)', flush=True)

print('[3/3] 保存 ...', flush=True)
feat_df = pd.DataFrame(rows)
feat_df.to_parquet(OUT)
print(f'已保存 {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB)', flush=True)

print('\n=== 各前缀口径样本量（存活人群 ∩ ≥2 行）===', flush=True)
for ptype in ['time', 'progress']:
    sub = feat_df[feat_df['prefix_type'] == ptype]
    for pv in sorted(sub['prefix_val'].unique()):
        s = sub[sub['prefix_val'] == pv]
        print(f'  {ptype}@{pv}{"min" if ptype=="time" else "%"}: 事务 {len(s):6,} '
              f'| 故障 {int(s["label"].sum()):5,} ({s["label"].mean()*100:5.2f}%) '
              f'| 启动型 {int((s["family"]=="startup").sum()):5,} 运行型 {int((s["family"]=="run").sum()):4,}', flush=True)
feat_cols = [c for c in feat_df.columns if c not in ('transaction_id', 'prefix_type', 'prefix_val',
             'n_prefix_rows', 'owner', 'family', 'label')]
print(f'\n特征数: {len(feat_cols)} | 特征列示例: {feat_cols[:8]} ...', flush=True)
print(f'总耗时 {time.time()-t0:.0f}s', flush=True)
