#!/usr/bin/env python3
"""build_prefix_seq_tensors.py — Phase 2 前缀序列张量构建（Token-Attn 前缀变体输入）
（gate_report_phase1 §4 冻结规格 / E11 / 与会议版同源对齐）

输入：
  data/prefix_dataset_full.parquet   —— 固化完整序列（vals_flat + offsets）
  data/prefix_feats_v1.parquet       —— 特征 v1（仅用其 cohort=tid 列表与 n_prefix_rows 做对齐断言）
输出：data/seq_tensors_tau{τ}.pkl   （τ∈{1,2,3,5,10,20}）
  X_tr/X_va/X_te : list[np (k_i,6)]  前缀段序列（已 per-prefix z-score）
  tids_*         : 事务 id 列表（与 X 同序，供特征对齐/归因回溯）
  y_tr/y_va/y_te : int64
  feat 由训练脚本按 tids 从 prefix_feats_v1 取（保证表侧/序列侧同一批事务）

关键设计：
  1. cohort 严格取 prefix_feats_v1 的 time@{τ} 行（=存活∩≥2行）→ 表侧与序列侧天然同序同集；
  2. 前缀段 = offsets ≤ τ 的行，断言行数 == prefix_feats_v1.n_prefix_rows（双重防错）；
  3. 【E11】归一化用 per-prefix z-score（只用前缀段自己的 mean/std），
     严禁沿用会议版 per-full-sequence 统计（那会偷看 τ 之后）；
  4. 切分与会议版同源：owner1-6 训练内 label 分层 20% val(seed42)，owner7-8 全量 test。
"""
import pandas as pd
import numpy as np
import pickle, os, json, time
from sklearn.model_selection import train_test_split

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, 'data')
FULL = os.path.join(DATA, 'prefix_dataset_full.parquet')
FEATV1 = os.path.join(DATA, 'prefix_feats_v1.parquet')
SCHEMA = json.load(open(os.path.join(DATA, 'prefix_features_v1.json')))

TIME_TAUS = [1, 2, 3, 5, 10, 20]
SEED = 42
CH = ['chargingv', 'charginga', 'out_power',
      'charging_gun_temperature1', 'charging_gun_temperature2', 'current_soc']

t0 = time.time()
print('[1/3] 载入固化数据集 + 特征 v1 cohort ...', flush=True)
full = pd.read_parquet(FULL, columns=['transaction_id', 'offsets', 'vals_flat', 'n_rows', 'owner'])
feat = pd.read_parquet(FEATV1, columns=['transaction_id', 'prefix_type', 'prefix_val',
                                        'n_prefix_rows', 'label', 'family', 'owner'])
# 建立 tid → 行的映射
full_index = full.set_index('transaction_id')

for tau in TIME_TAUS:
    cohort = feat[(feat['prefix_type'] == 'time') & (feat['prefix_val'] == tau)].reset_index(drop=True)
    tids = cohort['transaction_id'].tolist()
    print(f'\nτ={tau:2d}min: cohort {len(tids):,} 事务 (故障 {int(cohort["label"].sum()):,})', flush=True)

    # 逐事务截前缀段（list 收集，避免大 DataFrame 逐行操作）
    seqs, err = [], 0
    for i, tid in enumerate(tids):
        r = full_index.loc[tid]
        offs = np.asarray(r['offsets'], np.float32)
        vals = np.asarray(r['vals_flat'], np.float32).reshape(int(r['n_rows']), 6)
        mask = offs <= tau + 1e-9
        blk = vals[mask]
        if len(blk) != int(cohort.loc[i, 'n_prefix_rows']):   # 双重防错断言
            err += 1
        seqs.append(blk)
        if (i + 1) % 8000 == 0:
            print(f'    截取 {i+1}/{len(tids)} ({time.time()-t0:.0f}s)', flush=True)
    assert err == 0, f'{err} 个事务前缀行数与特征 v1 不一致 —— 管线断裂禁止继续'
    lens = np.array([len(s) for s in seqs])
    print(f'    前缀行数: min={lens.min()} p50={np.median(lens):.0f} max={lens.max()}', flush=True)

    # per-prefix z-score（只用前缀段统计 → E11 干净）
    seqs_z = []
    for s in seqs:
        z = s.astype(np.float64).copy()
        mu, sd = z.mean(0), z.std(0)
        sd = np.where(sd < 1e-6, 1.0, sd)
        z = (z - mu) / sd
        seqs_z.append(z.astype(np.float32))

    # 切分（与会议版 build_seq_tensors 同源：owner1-6 stratify 20% val, seed42；owner7-8 test）
    tr_owner = cohort['owner'].isin([f'Sheet{i}' for i in range(1, 7)])
    tr_idx = np.where(tr_owner.values)[0]
    te_idx = np.where(~tr_owner.values)[0]
    ytr_all = cohort['label'].values[tr_idx]
    i_tr, i_va = train_test_split(np.arange(len(tr_idx)), test_size=0.2,
                                  stratify=ytr_all, random_state=SEED)
    tr_sel, va_sel = tr_idx[i_tr], tr_idx[i_va]

    out = {
        'X_tr': [seqs_z[i] for i in tr_sel], 'tids_tr': [tids[i] for i in tr_sel],
        'y_tr': cohort['label'].values[tr_sel].astype(np.int64),
        'X_va': [seqs_z[i] for i in va_sel], 'tids_va': [tids[i] for i in va_sel],
        'y_va': cohort['label'].values[va_sel].astype(np.int64),
        'X_te': [seqs_z[i] for i in te_idx], 'tids_te': [tids[i] for i in te_idx],
        'y_te': cohort['label'].values[te_idx].astype(np.int64),
        'owner_te': cohort['owner'].values[te_idx],
        'feats': CH, 'tau_min': tau, 'seed': SEED,
        'n_train': len(tr_sel), 'n_val': len(va_sel), 'n_test': len(te_idx),
        'fault_train': int(cohort['label'].values[tr_sel].sum()),
        'fault_val': int(cohort['label'].values[va_sel].sum()),
        'fault_test': int(cohort['label'].values[te_idx].sum()),
    }
    p = os.path.join(DATA, f'seq_tensors_tau{tau}.pkl')
    with open(p, 'wb') as fp:
        pickle.dump(out, fp)
    print(f'    保存 {p} | train {len(tr_sel):,}(故障{out["fault_train"]:,}) '
          f'val {len(va_sel):,}({out["fault_val"]}) test {len(te_idx):,}({out["fault_test"]})', flush=True)

print(f'\n全部完成, 总耗时 {time.time()-t0:.0f}s', flush=True)
