#!/usr/bin/env python3
"""Enhanced feature engineering for improved fault recall.
Root cause: FN faults are "high-temperature overtemperature disconnects"
 (higher temps + higher SOC + longer duration) — look like normal charging.
Solution: add temporal change-rate features (temp slope, current surge, voltage sag).
"""
import pandas as pd, numpy as np, time, os, warnings
warnings.filterwarnings('ignore')
import openpyxl

DATA = '/Users/arthas/git/excharge/data/real/'
OUT  = '/Users/arthas/git/excharge/docs/'

t0 = time.time()
df = pd.read_parquet(f'{DATA}/all_data.parquet')
print(f'Loaded {len(df):,} rows in {time.time()-t0:.0f}s', flush=True)

# Sequence-level label & owner (from earlier run, already computed)
tx_label = df.groupby('transaction_id')['label'].first()
tx_owner = df.groupby('transaction_id')['owner'].first()

seq_len = df.groupby('transaction_id').size()
valid_tx = seq_len[seq_len >= 30].index

# Owner split
train_owners = ['Sheet1','Sheet2','Sheet3','Sheet4','Sheet5','Sheet6']
test_owners  = ['Sheet7','Sheet8']
train_tx = [t for t in valid_tx if tx_owner[t] in train_owners]
test_tx  = [t for t in valid_tx if tx_owner[t] in test_owners]

print(f'Train: {len(train_tx):,} | Test: {len(test_tx):,}', flush=True)

# ---- Enhanced feature engineering ----
def enhanced_features(sub):
    """Extract enhanced sequence-level features, including temporal dynamics."""
    v = sub['chargingv'].values.astype(np.float32)
    a = sub['charginga'].values.astype(np.float32)
    p = sub['out_power'].values.astype(np.float32)
    soc = sub['current_soc'].values.astype(np.float32)
    t1 = sub['charging_gun_temperature1'].values.astype(np.float32)
    t2 = sub['charging_gun_temperature2'].values.astype(np.float32)

    n = len(v)
    if n == 0:
        return {}

    feat = {}

    # === Basic stats (from Route B) ===
    feat['v_mean'] = v.mean(); feat['v_std'] = v.std()
    feat['v_min'] = v.min(); feat['v_max'] = v.max()
    feat['v_first'] = v[0]; feat['v_last'] = v[-1]
    feat['v_slope'] = (v[-1]-v[0])/(n-1) if n>1 else 0
    feat['a_mean'] = a.mean(); feat['a_std'] = a.std()
    feat['a_min'] = a.min(); feat['a_max'] = a.max()
    feat['a_first'] = a[0]; feat['a_last'] = a[-1]
    feat['p_mean'] = p.mean(); feat['p_std'] = p.std()
    feat['p_max'] = p.max(); feat['p_min'] = p.min()
    feat['p_first'] = p[0]; feat['p_last'] = p[-1]
    feat['soc_first'] = soc[0] if not np.isnan(soc[0]) else np.nan
    feat['soc_last'] = soc[-1] if not np.isnan(soc[-1]) else np.nan
    feat['soc_delta'] = feat['soc_last'] - feat['soc_first'] if (not np.isnan(soc[0]) and not np.isnan(soc[-1])) else np.nan
    feat['t1_mean'] = np.nanmean(t1); feat['t1_max'] = np.nanmax(t1)
    feat['t2_mean'] = np.nanmean(t2); feat['t2_max'] = np.nanmax(t2)
    feat['t1_last'] = t1[-1] if not np.isnan(t1[-1]) else np.nan
    feat['t2_last'] = t2[-1] if not np.isnan(t2[-1]) else np.nan
    feat['n_points'] = n
    feat['duration_min'] = sub['total_charging_min'].max()
    feat['total_kwh'] = sub['total_charging_kwh'].max()
    feat['p_v_ratio'] = feat['p_mean']/feat['v_mean'] if feat['v_mean'] > 0 else np.nan

    # === NEW: Temporal dynamics features ===
    # Temperature change rate (critical for detecting overtemperature trend)
    if n >= 2:
        t1_valid = t1[~np.isnan(t1)]
        t2_valid = t2[~np.isnan(t2)]
        feat['t1_slope'] = (t1[-1]-t1[0])/(n-1) if len(t1_valid)>=2 else 0
        feat['t2_slope'] = (t2[-1]-t2[0])/(n-1) if len(t2_valid)>=2 else 0
        # Max temp rise rate (max single-step jump)
        t1_diff = np.diff(t1)
        feat['t1_max_jump'] = np.nanmax(np.abs(t1_diff))
        t2_diff = np.diff(t2)
        feat['t2_max_jump'] = np.nanmax(np.abs(t2_diff))
        # Temp volatility (rolling std in 2nd half — charging peak phase)
        half = n // 2
        feat['t1_std_2nd'] = np.nanstd(t1[half:]) if n-half > 1 else 0
        feat['t2_std_2nd'] = np.nanstd(t2[half:]) if n-half > 1 else 0
        # Voltage sag: min voltage in first 1/3 (sag at start indicates problems)
        first_third = v[:max(1, n//3)]
        feat['v_first_third_min'] = first_third.min()
        feat['v_sag_from_mean'] = feat['v_mean'] - first_third.min()
        # Current surge: max current in last 1/3 (surge before disconnect)
        last_third = a[max(0, 2*n//3):]
        feat['a_last_third_max'] = last_third.max()
        feat['a_first_last_ratio'] = a[-1]/a[0] if a[0] > 0 else np.nan  # disconnect pattern
        # Power volatility
        feat['p_std'] = p.std()
        feat['p_max_jump'] = np.nanmax(np.abs(np.diff(p)))
        # SOC trajectory
        soc_diff = np.diff(soc)
        feat['soc_rate'] = soc_diff.mean() if len(soc_diff) > 0 else 0
        feat['soc_last_rate'] = soc_diff[-1] if len(soc_diff) > 0 else 0  # final SOC rate
    else:
        for k in ['t1_slope','t2_slope','t1_max_jump','t2_max_jump',
                  't1_std_2nd','t2_std_2nd','v_first_third_min','v_sag_from_mean',
                  'a_last_third_max','a_first_last_ratio','p_std','p_max_jump',
                  'soc_rate','soc_last_rate']:
            feat[k] = 0

    # === NEW: Segment-based features ===
    if n >= 6:
        seg = n // 3
        v_seg1 = v[:seg]; v_seg2 = v[seg:2*seg]; v_seg3 = v[2*seg:]
        a_seg1 = a[:seg]; a_seg2 = a[seg:2*seg]; a_seg3 = a[2*seg:]
        feat['v_seg_change_1to2'] = v_seg2.mean() - v_seg1.mean()
        feat['v_seg_change_2to3'] = v_seg3.mean() - v_seg2.mean()
        feat['a_seg_change_1to2'] = a_seg2.mean() - a_seg1.mean()
        feat['a_seg_change_2to3'] = a_seg3.mean() - a_seg2.mean()
        feat['a_seg3_max'] = a_seg3.max()
        # Detect abrupt disconnect: last 3 points vs first 3 points
        feat['v_last3_vs_first3'] = v[-3:].mean() - v[:3].mean()
        feat['a_last3_vs_first3'] = a[-3:].mean() - a[:3].mean()
        feat['p_last3_vs_first3'] = p[-3:].mean() - p[:3].mean()
    else:
        for k in ['v_seg_change_1to2','v_seg_change_2to3','a_seg_change_1to2','a_seg_change_2to3',
                  'a_seg3_max','v_last3_vs_first3','a_last3_vs_first3','p_last3_vs_first3']:
            feat[k] = 0

    # === NEW: Overtemperature indicator ===
    feat['t1_over_40'] = int(np.nanmax(t1) > 40)
    feat['t2_over_40'] = int(np.nanmax(t2) > 40)
    feat['t1_over_45'] = int(np.nanmax(t1) > 45)
    feat['t2_over_45'] = int(np.nanmax(t2) > 45)

    return feat

print('Extracting enhanced features...', flush=True)
t1 = time.time()

records = []
labels = []
owners = []
for i, tx in enumerate(valid_tx):
    sub = df[df['transaction_id'] == tx].sort_values('end_time')
    f = enhanced_features(sub)
    if f:
        records.append(f)
        labels.append(tx_label[tx])
        owners.append(tx_owner[tx])
    if (i+1) % 5000 == 0:
        print(f'  {i+1}/{len(valid_tx)}', flush=True)

seq_df = pd.DataFrame(records)
seq_df['label'] = labels
seq_df['owner'] = owners
print(f'Built {len(seq_df):,} sequences with {len(seq_df.columns)} features in {time.time()-t1:.0f}s', flush=True)

# Align feature columns
all_feats = [c for c in seq_df.columns if c not in ('label','owner')]
print(f'Features ({len(all_feats)}): {all_feats}')

# Battery type from raw data (add separately)
bt = df.groupby('transaction_id')['types'].first()
bt_map = {3:'LFP',6:'NMC',4:'LMO',5:'LCO',7:'LP'}
bt_labels = bt[seq_df.index.map(lambda x: valid_tx[x] if x < len(list(valid_tx)) else x)]
seq_df['battery_type'] = seq_df.index.map(lambda i: bt_map.get(bt.get(valid_tx[i] if i < len(valid_tx) else i, 0), 'UNK'))
bt_dummies = pd.get_dummies(seq_df['battery_type'], prefix='bt')
seq_df = pd.concat([seq_df.drop(columns=['battery_type']), bt_dummies], axis=1)

# Split
train_mask = seq_df['owner'].isin(train_owners)
test_mask  = seq_df['owner'].isin(test_owners)
train_df = seq_df[train_mask].copy()
test_df  = seq_df[test_mask].copy()

feat_cols = [c for c in seq_df.columns if c not in ('label','owner')]

X_train = train_df[feat_cols].fillna(train_df[feat_cols].median())
X_test  = test_df[feat_cols].fillna(train_df[feat_cols].median())  # use train median
y_train = train_df['label'].values
y_test  = test_df['label'].values

print(f'Train: {len(X_train):,} (fault {y_train.sum():,})')
print(f'Test:  {len(X_test):,} (fault {y_test.sum():,})')

# Save
X_train.to_parquet(f'{DATA}/seq_X_train_v2.parquet')
X_test.to_parquet(f'{DATA}/seq_X_test_v2.parquet')
pd.DataFrame({'label': y_train}).to_parquet(f'{DATA}/seq_y_train_v2.parquet')
pd.DataFrame({'label': y_test}).to_parquet(f'{DATA}/seq_y_test_v2.parquet')
import json
json.dump({'features': feat_cols, 'n_train': len(X_train), 'n_test': len(X_test),
           'n_features': len(feat_cols)},
          open(f'{DATA}/seq_features_v2.json','w'))
print(f'\nSaved v2 features ({len(feat_cols)} cols) in {time.time()-t0:.0f}s total', flush=True)
print('New features:', [c for c in feat_cols if c not in [
    'n_points','duration_min','total_kwh','v_mean','v_std','v_min','v_max','v_first','v_last','v_slope',
    'a_mean','a_std','a_min','a_max','a_first','a_last',
    'p_mean','p_std','p_max','p_min','p_first','p_last',
    'soc_first','soc_last','soc_delta',
    't1_mean','t1_max','t2_mean','t2_max','t1_last','t2_last',
    'p_v_ratio','bt_LFP','bt_NMC','bt_LMO','bt_LCO','bt_LP']])
