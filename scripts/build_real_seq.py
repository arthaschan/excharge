#!/usr/bin/env python3
"""Route B: Build sequence-level dataset from Nature dataset.
- Split transactions into train/val/test by owner (federated-like split)
- Extract per-transaction statistics (seq-level features for XGBoost baseline)
- Save windowed sequences (for Transformer baseline)
"""
import pandas as pd, numpy as np, json, os, time

DATA = '/Users/arthas/git/excharge/data/real/'
os.makedirs(DATA, exist_ok=True)

t0 = time.time()
df = pd.read_parquet(f'{DATA}/all_data.parquet')
print(f'Loaded {len(df):,} rows in {time.time()-t0:.0f}s', flush=True)

# --- Battery type mapping (from Readme: 03=LFP, 06=NMC, 04=LMO, 05=LCO, 07=LP) ---
type_map = {3: 'LFP', 6: 'NMC', 4: 'LMO', 5: 'LCO', 7: 'LP'}
df['battery_type'] = df['types'].map(type_map)

# --- Sequence-level aggregation ---
print('Aggregating sequences...', flush=True)
t1 = time.time()

def agg_seq(g):
    v = g['chargingv'].values
    a = g['charginga'].values
    p = g['out_power'].values
    soc = g['current_soc'].values
    t1_ = g['charging_gun_temperature1'].values
    t2_ = g['charging_gun_temperature2'].values
    return pd.Series({
        'owner': g['owner'].iloc[0],
        'label': g['label'].iloc[0],
        'battery_type': g['battery_type'].iloc[0],
        'n_points': len(g),
        'duration_min': g['total_charging_min'].max(),
        'total_kwh': g['total_charging_kwh'].max(),
        # voltage stats
        'v_mean': v.mean(), 'v_std': v.std(), 'v_min': v.min(), 'v_max': v.max(),
        'v_first': v[0] if len(v) else np.nan, 'v_last': v[-1] if len(v) else np.nan,
        'v_slope': (v[-1]-v[0]) if len(v) > 1 else 0,
        # current stats
        'a_mean': a.mean(), 'a_std': a.std(), 'a_min': a.min(), 'a_max': a.max(),
        'a_first': a[0] if len(a) else np.nan, 'a_last': a[-1] if len(a) else np.nan,
        # power stats
        'p_mean': p.mean(), 'p_std': p.std(), 'p_max': p.max(), 'p_min': p.min(),
        'p_first': p[0] if len(p) else np.nan, 'p_last': p[-1] if len(p) else np.nan,
        # SOC stats
        'soc_first': soc[0] if len(soc) else np.nan,
        'soc_last': soc[-1] if len(soc) else np.nan,
        'soc_delta': (soc[-1]-soc[0]) if len(soc) > 1 else 0,
        # temperature
        't1_mean': np.nanmean(t1_), 't1_max': np.nanmax(t1_) if len(t1_) else np.nan,
        't2_mean': np.nanmean(t2_), 't2_max': np.nanmax(t2_) if len(t2_) else np.nan,
        't1_last': t1_[-1] if len(t1_) else np.nan,
        't2_last': t2_[-1] if len(t2_) else np.nan,
        # derived
        'p_v_ratio': (p.mean()/v.mean()) if len(v) and v.mean() > 0 else np.nan,
    })

seq = df.groupby('transaction_id', group_keys=False).apply(agg_seq)
seq.index.name = 'transaction_id'
seq = seq.reset_index()
print(f'Sequences: {len(seq):,} in {time.time()-t1:.0f}s', flush=True)

# --- Only sequences >= 30 points (matching Nature paper) ---
seq30 = seq[seq['n_points'] >= 30].copy()
print(f'Sequences >= 30 points: {len(seq30):,}', flush=True)

# --- Split: owners 1-6 train/val, 7-8 test (new owners, matching paper) ---
# owner names: Sheet1..Sheet8
train_owners = ['Sheet1','Sheet2','Sheet3','Sheet4','Sheet5','Sheet6']
test_owners = ['Sheet7','Sheet8']

train = seq30[seq30['owner'].isin(train_owners[:4])]
val = seq30[seq30['owner'].isin(train_owners[4:])]
test = seq30[seq30['owner'].isin(test_owners)]

print(f'\nSplit:')
print(f'  Train (owners 1-4): {len(train):,} (fault {train.label.sum():,})')
print(f'  Val   (owners 5-6): {len(val):,} (fault {val.label.sum():,})')
print(f'  Test  (owners 7-8): {len(test):,} (fault {test.label.sum():,})')

# --- Feature columns ---
feat_cols = ['n_points','duration_min','total_kwh',
             'v_mean','v_std','v_min','v_max','v_first','v_last','v_slope',
             'a_mean','a_std','a_min','a_max','a_first','a_last',
             'p_mean','p_std','p_max','p_min','p_first','p_last',
             'soc_first','soc_last','soc_delta',
             't1_mean','t1_max','t2_mean','t2_max','t1_last','t2_last',
             'p_v_ratio']
# one-hot battery type
bt = pd.get_dummies(seq30['battery_type'], prefix='bt')
X = pd.concat([seq30[feat_cols].reset_index(drop=True), bt.reset_index(drop=True)], axis=1)
X = X.fillna(X.median())

print(f'\nFeatures: {X.shape[1]}')

# Save
train_idx = seq30.index.isin(train.index)
val_idx = seq30.index.isin(val.index)
test_idx = seq30.index.isin(test.index)

X_train, y_train = X[train_idx], seq30['label'][train_idx]
X_val, y_val = X[val_idx], seq30['label'][val_idx]
X_test, y_test = X[test_idx], seq30['label'][test_idx]

X_train.to_parquet(f'{DATA}/seq_X_train.parquet')
X_val.to_parquet(f'{DATA}/seq_X_val.parquet')
X_test.to_parquet(f'{DATA}/seq_X_test.parquet')
pd.DataFrame({'label': y_train.values}).to_parquet(f'{DATA}/seq_y_train.parquet')
pd.DataFrame({'label': y_val.values}).to_parquet(f'{DATA}/seq_y_val.parquet')
pd.DataFrame({'label': y_test.values}).to_parquet(f'{DATA}/seq_y_test.parquet')

json.dump({'features': list(X.columns), 'n_train': len(X_train), 'n_val': len(X_val), 'n_test': len(X_test)},
          open(f'{DATA}/seq_features.json','w'), indent=2)

print(f'\nSaved sequence features. Total time {time.time()-t0:.0f}s', flush=True)
