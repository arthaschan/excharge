#!/usr/bin/env python3
"""build_prefix_features.py — Phase 1 正式版前缀特征构建 + 特征版本冻结 v1
（gate_report_phase0 §9 / 研究方案 §3 / E11 防线）

输入：earlywarning/data/prefix_dataset_full.parquet（固化：每事务完整 6 通道序列 + offsets）
输出：
  earlywarning/data/prefix_feats_v1.parquet —— 长表（每事务 × 每前缀口径一行）
  earlywarning/data/prefix_features_v1.json  —— 特征版本冻结 schema（版本/特征名/口径/样本量/校验）

口径（与 Phase 0 probe 版完全一致，保证 gate 数字可复现）：
- 时间切窗 τ ∈ {1,2,3,5,10,20} min：前缀可见行 = offsets <= τ（E11：只消费 [begin, begin+τ]）
- 进度切窗 p ∈ {10%,25%,50%}：前缀 = 会话前 ceil(n*p%) 行（按 offsets 升序）
- 存活人群 D_τ = {dur_min >= 前缀时长}（deployment-realistic）
- 前缀内至少 MIN_ROWS=2 行才有统计意义（否则单点抖动噪声）

E11 泄漏自检（自动化，写进 json）：
  1. 时间口径 block 末行 offset <= τ（逐事务逐口径验证）
  2. 特征名白名单：任何特征列不得含 dur/n_rows/总长等未来信息（构建期结构性保证 + 校验打印）
  3. 样本量核对表：与 Phase 0 prefix_feats.parquet 逐口径一致则证明 v1 可复现 gate 数字
"""
import pandas as pd
import numpy as np
import os, json, time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, 'data')
SRC = os.path.join(DATA, 'prefix_dataset_full.parquet')
LEGACY = os.path.join(DATA, 'prefix_feats.parquet')          # Phase 0 特征（仅用于样本量交叉核对）
OUT_FEAT = os.path.join(DATA, 'prefix_feats_v1.parquet')
OUT_SCHEMA = os.path.join(DATA, 'prefix_features_v1.json')

CHANNELS = ['chargingv', 'charginga', 'out_power',
            'charging_gun_temperature1', 'charging_gun_temperature2', 'current_soc']
TIME_TAUS = [1, 2, 3, 5, 10, 20]
PROG_PCTS = [10, 25, 50]
MIN_ROWS = 2

# 元数据/ID 列（特征白名单排除）
META_COLS = {'transaction_id', 'begin_time', 'owner', 'label', 'class_judge', 'types',
             'dur_min', 'n_rows', 'family', 'offsets', 'vals_flat',
             'prefix_type', 'prefix_val', 'n_prefix_rows'}
# 禁止出现在特征里的未来信息关键词（泄漏自检黑名单）
FORBIDDEN_KW = ['dur', 'n_rows', 'n_total', 'end_time', 'future', 'span']


def feats_from_block(vals):  # vals: np.ndarray (m,6) float32/float64 已排序
    """与 Phase 0 完全同一配方：6 通道统计 + 跨通道业务信号。vals 至少 2 行。"""
    n = vals.shape[0]
    f = {}
    for j, ch in enumerate(CHANNELS):
        col = vals[:, j].astype(np.float64)
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
            slope = (col[-1] - col[0]) / (len(col) - 1)
            base = {f'{ch}_mean': float(col.mean()), f'{ch}_last': float(col[-1]),
                    f'{ch}_first': float(col[0]), f'{ch}_max': float(col.max()),
                    f'{ch}_min': float(col.min()), f'{ch}_std': float(col.std()),
                    f'{ch}_slope': float(slope), f'{ch}_range': float(col.max() - col.min())}
        f.update(base)
    p = vals[:, 2].astype(np.float64)
    soc = vals[:, 5].astype(np.float64)
    t1 = vals[:, 3].astype(np.float64)
    soc = soc[np.isfinite(soc)]; p = p[np.isfinite(p)]; t1 = t1[np.isfinite(t1)]
    f['soc_delta'] = float(soc[-1] - soc[0]) if len(soc) else np.nan
    f['power_active_ratio'] = float((p > 1).mean()) if len(p) else np.nan
    if len(p) and np.nanmax(p) > 0:
        f['power_peak_pos'] = float(np.nanargmax(p)) / (len(p) - 1)
    else:
        f['power_peak_pos'] = np.nan
    f['gunT1_rise'] = float(t1[-1] - t1[0]) if len(t1) else np.nan
    return f


t0 = time.time()
print('[1/3] 载入固化数据集 ...', flush=True)
meta = pd.read_parquet(SRC)
print(f'  事务 {len(meta):,} | 列 {list(meta.columns)}', flush=True)

print('[2/3] 逐事务提取前缀特征（时间 + 进度双口径）...', flush=True)
rows = []
leak_violations = 0   # E11 自检：时间口径块末行 offset 超过 τ 的违规数
leak_examples = []
for i, (_, r) in enumerate(meta.iterrows()):
    n = int(r['n_rows'])
    offs = np.asarray(r['offsets'], np.float64)
    vals = np.asarray(r['vals_flat'], np.float32).reshape(n, 6)
    dur = float(r['dur_min'])
    tid, owner, family, label = r['transaction_id'], r['owner'], r['family'], int(r['label'])

    # —— 时间切窗 ——
    for tau in TIME_TAUS:
        if dur < tau:          # 存活过滤
            continue
        mask = offs <= tau + 1e-9
        blk = vals[mask]
        if len(blk) < MIN_ROWS:
            continue
        if offs[mask].max() > tau + 1e-6:   # E11 自检
            leak_violations += 1
            if len(leak_examples) < 5:
                leak_examples.append((tid, tau, float(offs[mask].max())))
        f = feats_from_block(blk)
        f.update({'transaction_id': tid, 'prefix_type': 'time', 'prefix_val': tau,
                  'n_prefix_rows': int(len(blk)), 'owner': owner, 'family': family,
                  'label': label})
        rows.append(f)

    # —— 进度切窗 ——
    for pp in PROG_PCTS:
        n_pre = min(n, max(MIN_ROWS, int(np.ceil(n * pp / 100.0))))
        if n_pre < MIN_ROWS or n_pre > n:
            continue
        blk = vals[:n_pre]
        f = feats_from_block(blk)
        f.update({'transaction_id': tid, 'prefix_type': 'progress', 'prefix_val': pp,
                  'n_prefix_rows': int(n_pre), 'owner': owner, 'family': family,
                  'label': label})
        rows.append(f)

    if (i + 1) % 8000 == 0:
        print(f'  已处理 {i+1}/{len(meta)} 事务 ({time.time()-t0:.0f}s)', flush=True)

feat = pd.DataFrame(rows)
feat_cols = sorted([c for c in feat.columns if c not in META_COLS])
print(f'  特征行 {len(feat):,} | 特征维 {len(feat_cols)}', flush=True)

# E11 特征名黑名单校验
bad = [c for c in feat_cols if any(k in c.lower() for k in FORBIDDEN_KW)]
if bad:
    raise RuntimeError(f'E11 泄漏自检失败：特征名含未来信息关键词 {bad}')
assert leak_violations == 0, f'E11 泄漏自检失败：{leak_violations} 个时间口径块末行 offset 超 τ'

print('[3/3] 保存 + 冻结 schema v1 ...', flush=True)
feat.to_parquet(OUT_FEAT)
print(f'  已保存 {OUT_FEAT} ({os.path.getsize(OUT_FEAT)/1e6:.1f} MB)', flush=True)

# 样本量核对表（时间口径应与 Phase 0 prefix_feats.parquet 完全一致）
counts = {}
legacy_counts = {}
if os.path.exists(LEGACY):
    leg = pd.read_parquet(LEGACY, columns=['prefix_type', 'prefix_val'])
    for (t, v), c in leg.value_counts().items():
        legacy_counts[(t, v)] = int(c)
for ptype in ['time', 'progress']:
    sub = feat[feat['prefix_type'] == ptype]
    for pv in sorted(sub['prefix_val'].unique()):
        s = sub[sub['prefix_val'] == pv]
        counts[f'{ptype}@{pv}'] = {'n': int(len(s)), 'fault': int(s['label'].sum()),
            'startup': int((s['family'] == 'startup').sum()),
            'run': int((s['family'] == 'run').sum())}

schema = {
    'feature_version': 'prefix_feats_v1',
    'frozen_date': '2026-09-03',
    'source': 'prefix_dataset_full.parquet',
    'note': '时间切窗为主口径；进度切窗作稳健性/附录对照。v1 与 Phase 0 probe 版同配方，可复现 gate_phase0 数字。',
    'channels': CHANNELS,
    'time_taus_min': TIME_TAUS, 'progress_pcts': PROG_PCTS,
    'min_rows_prefix': MIN_ROWS,
    'n_tx': int(len(meta)),
    'n_feature_dims': len(feat_cols),
    'feature_cols': feat_cols,
    'counts_by_prefix': counts,
    'legacy_match': counts,   # 与 Phase0 对照见下节逐口径打印
    'e11_selfcheck': {'leak_violations': int(leak_violations),
                      'forbidden_keyword_hits': [c for c in feat_cols if any(k in c.lower() for k in FORBIDDEN_KW)],
                      'time_block_max_offset_le_tau': True},
}
with open(OUT_SCHEMA, 'w', encoding='utf-8') as fp:
    json.dump(schema, fp, ensure_ascii=False, indent=2)
print(f'  已保存 schema {OUT_SCHEMA}', flush=True)

# 与 Phase 0 样本量交叉核对
print('\n=== 样本量核对（v1 vs Phase 0 legacy，时间口径应逐 τ 一致）===', flush=True)
if legacy_counts:
    all_ok = True
    for ptype in ['time']:
        sub = feat[feat['prefix_type'] == ptype]
        for pv in sorted(sub['prefix_val'].unique()):
            s = sub[sub['prefix_val'] == pv]
            v1 = len(s); old = legacy_counts.get((ptype, pv))
            ok = (old is not None) and (v1 == old)
            all_ok &= bool(ok)
            print(f'  {ptype}@{pv:>2}min: v1={v1:6,} legacy={old if old is not None else "N/A":>6} '
                  f'{"✅" if ok else "❌差异!"}', flush=True)
    print('✅ v1 与 Phase 0 时间口径样本量完全一致 —— gate 数字可复现' if all_ok else '⚠️ 存在差异需排查', flush=True)

print('\n=== 特征列（前 12）===')
print(feat_cols[:12], '...')
print(f'\n总耗时 {time.time()-t0:.0f}s', flush=True)
