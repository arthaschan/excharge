#!/usr/bin/env python3
"""build_prefix_dataset.py — Phase 1 正式版数据集构建（gate_report_phase0 §9 + 研究方案 §3 定稿口径）

输入：/Users/arthas/git/excharge/data/real/all_data.parquet（155.6万采样点，31,449 事务）
输出：
  earlywarning/data/prefix_dataset_full.parquet —— 每事务一行（固化完整序列，深度模型任意 τ 现场截段）
  earlywarning/data/split.json                    —— 跨站统一切分（owner1-6 训 → owner7-8 测，E11-2 口径固化）

设计要点（相对 probe 版的关键升级）：
  1. 【序列张量】每事务保存完整 6 通道序列 vals(n,6,float32) + 行级偏移 offsets(n,)(距 begin 的分钟)
     → Phase 2 深度模型按 τ 截段 [offsets<=τ] + length mask，无需重读 111MB 原始文件；
  2. 【E11 防泄漏】特征/张量只消费 offsets<=τ 的行；offsets 单调性内置自检；
  3. 【口径固化】时间语义实测确认：begin_time=会话开始(事务内恒定)、end_time=行级采样戳(+60s)；
     双谱系阈值 30min；family∈{startup,run,normal} 与 Phase 0 完全一致。
"""

import pandas as pd
import numpy as np
import os, json, time

SRC = '/Users/arthas/git/excharge/data/real/all_data.parquet'
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, 'data')
OUT_PQ = os.path.join(OUT_DIR, 'prefix_dataset_full.parquet')
OUT_SPLIT = os.path.join(OUT_DIR, 'split.json')

CHANNELS = ['chargingv', 'charginga', 'out_power',
            'charging_gun_temperature1', 'charging_gun_temperature2', 'current_soc']
FAULT_THRESH_MIN = 30  # startup <30min, run >=30min
TRAIN_OWNERS = [f'Sheet{i}' for i in range(1, 7)]
TEST_OWNERS = ['Sheet7', 'Sheet8']

t0 = time.time()
print('[1/4] 读取 all_data.parquet ...', flush=True)
df = pd.read_parquet(SRC)
df['begin_time'] = pd.to_datetime(df['begin_time'])
df['end_time'] = pd.to_datetime(df['end_time'])
print(f'  原始行 {len(df):,} | 事务 {df["transaction_id"].nunique():,}', flush=True)

print('[2/4] 逐事务聚合为完整序列（vals + offsets + 元数据）...', flush=True)
df = df.sort_values(['transaction_id', 'end_time']).reset_index(drop=True)

tids = df['transaction_id'].unique()
rows = []
n_dup_ts = 0      # 同行时间戳重复计数（同 offset 出现 >1 行的事务）
n_nan_rows = 0    # 含 NaN 的行数（记录以便特征脚本决策）
for i, tid in enumerate(tids):
    sub = df[df['transaction_id'] == tid]
    vals = sub[CHANNELS].to_numpy(dtype=np.float32)              # (n,6)
    offs = (sub['end_time'] - sub['begin_time'].iloc[0]).dt.total_seconds().to_numpy(dtype=np.float32) / 60.0
    if np.any(np.diff(offs) < 0):
        raise RuntimeError(f'事务 {tid} offsets 非单调 —— 排序或时间语义错误，禁止继续')
    if len(np.unique(np.round(offs, 3))) < len(offs):
        n_dup_ts += 1
    if np.isnan(vals).any():
        n_nan_rows += int(np.isnan(vals).any(axis=1).sum())
    rows.append({
        'transaction_id': tid,
        'begin_time': sub['begin_time'].iloc[0],
        'owner': sub['owner'].iloc[0],
        'label': int(sub['label'].max()),
        'class_judge': int(sub['class_judge'].mode().iloc[0]) if len(sub) else -1,
        'types': int(sub['types'].mode().iloc[0]) if len(sub) else -1,
        'dur_min': float(offs.max()),
        'n_rows': int(len(sub)),
        'offsets': offs,
        'vals': vals,
    })
    if (i + 1) % 8000 == 0:
        print(f'  已聚合 {i+1}/{len(tids)} 事务 ({time.time()-t0:.0f}s)', flush=True)

meta = pd.DataFrame(rows)
meta['family'] = np.where(meta['label'] == 0, 'normal',
                  np.where(meta['dur_min'] < FAULT_THRESH_MIN, 'startup', 'run'))
print(f'  谱系分布: {meta["family"].value_counts().to_dict()}', flush=True)
print(f'  含重复时间戳的事务数: {n_dup_ts} | 含 NaN 通道值行数: {n_nan_rows:,}', flush=True)
print(f'  每事务行数 min/med/max: {meta["n_rows"].min()}/{meta["n_rows"].median():.0f}/{meta["n_rows"].max()}', flush=True)

print('[3/4] 保存固化数据集 + 跨站 split.json ...', flush=True)
os.makedirs(OUT_DIR, exist_ok=True)
# vals 二维 (n,6) 展平为 (n*6,) 一维 + 显式 list<float32> 类型（pyarrow 拒绝自动推断 2-D list）
import pyarrow as pa
import pyarrow.parquet as pq
tbl = pa.Table.from_pandas(meta.drop(columns=['vals', 'offsets']))
offsets_arr = pa.array([np.asarray(v, dtype=np.float32) for v in meta['offsets']],
                       type=pa.list_(pa.float32()))
vals_flat_arr = pa.array([np.asarray(v, dtype=np.float32).reshape(-1) for v in meta['vals']],
                         type=pa.list_(pa.float32()))
tbl = tbl.append_column('offsets', offsets_arr).append_column('vals_flat', vals_flat_arr)
pq.write_table(tbl, OUT_PQ)
print(f'  已保存 {OUT_PQ} ({os.path.getsize(OUT_PQ)/1e6:.1f} MB)', flush=True)
print('  注: vals_flat = 每事务完整 6 通道展平 (n*6,) float32, 读回时 reshape(-1,6)', flush=True)

split = {
    'protocol': 'cross_station_owner1_6_train_to_7_8_test',
    'note': '主口径=跨站冷启动（与会议论文一致）；同分布随机分层作为方法学内参由实验脚本按 seed 现场切分。',
    'train_owners': TRAIN_OWNERS, 'test_owners': TEST_OWNERS,
    'n_train_tx': int(meta['owner'].isin(TRAIN_OWNERS).sum()),
    'n_test_tx': int(meta['owner'].isin(TEST_OWNERS).sum()),
    'per_owner': {ow: {'n': int((meta['owner'] == ow).sum()),
                       'fault': int(((meta['owner'] == ow) & (meta['label'] == 1)).sum())}
                  for ow in TRAIN_OWNERS + TEST_OWNERS},
}
with open(OUT_SPLIT, 'w', encoding='utf-8') as fp:
    json.dump(split, fp, ensure_ascii=False, indent=2)
print(f'  已保存 {OUT_SPLIT}', flush=True)

print('[4/4] 事务级口径核对表 ...', flush=True)
print('\n  owner × family:')
print(pd.crosstab(meta['owner'], meta['family']).to_string(), flush=True)
n_train = split['n_train_tx']; n_test = split['n_test_tx']
print(f'\n  训练域(owner1-6): {n_train:,} 事务 | 测试域(owner7-8): {n_test:,} 事务', flush=True)
print(f'  全量故障率: {meta["label"].mean()*100:.2f}% | startup {int((meta["family"]=="startup").sum())} '
      f'/ run {int((meta["family"]=="run").sum())}', flush=True)
print(f'\n总耗时 {time.time()-t0:.0f}s', flush=True)
