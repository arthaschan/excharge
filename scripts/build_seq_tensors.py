#!/usr/bin/env python3
"""构建序列级张量：每条充电序列(>=30点) -> 变长时序张量 [L, 6]。
特征通道: chargingv, charginga, out_power, charging_gun_temperature1,
          charging_gun_temperature2, current_soc
每序列按特征 z-score 归一化。保存为 pickle(变长 list)。
划分: Train=owner1-6, Test=owner7-8(新站点跨域)。
"""
import pandas as pd, numpy as np, pickle, time, os, warnings
warnings.filterwarnings('ignore')

DATA = '/Users/arthas/git/excharge/data/real/'
OUT = '/Users/arthas/git/excharge/data/real/'
SEED = 42
np.random.seed(SEED)

t0 = time.time()
df = pd.read_parquet(f'{DATA}/all_data.parquet')
print(f'Loaded {len(df):,} rows in {time.time()-t0:.0f}s', flush=True)

FEATS = ['chargingv', 'charginga', 'out_power',
         'charging_gun_temperature1', 'charging_gun_temperature2', 'current_soc']

tx_label = df.groupby('transaction_id')['label'].first()
tx_owner = df.groupby('transaction_id')['owner'].first()

# 过滤 >= 30 点
seq_len = df.groupby('transaction_id').size()
valid_tx = seq_len[seq_len >= 30].index
print(f'Valid transactions (>=30 pts): {len(valid_tx):,}', flush=True)

train_owners = ['Sheet1','Sheet2','Sheet3','Sheet4','Sheet5','Sheet6']
test_owners  = ['Sheet7','Sheet8']

train_tx = [t for t in valid_tx if tx_owner[t] in train_owners]
test_tx  = [t for t in valid_tx if tx_owner[t] in test_owners]
print(f'Train tx: {len(train_tx):,} (fault {sum(tx_label[t] for t in train_tx):,})', flush=True)
print(f'Test tx:  {len(test_tx):,} (fault {sum(tx_label[t] for t in test_tx):,})', flush=True)

# 分层切 val
labels_arr = np.array([tx_label[t] for t in train_tx])
from sklearn.model_selection import train_test_split
tr_idx, va_idx = train_test_split(np.arange(len(train_tx)), test_size=0.2,
                                  stratify=labels_arr, random_state=SEED)
tr_tx = [train_tx[i] for i in tr_idx]
va_tx = [train_tx[i] for i in va_idx]
print(f'  Train: {len(tr_tx):,} (fault {sum(tx_label[t] for t in tr_tx):,})', flush=True)
print(f'  Val:   {len(va_tx):,} (fault {sum(tx_label[t] for t in va_tx):,})', flush=True)

def build(tx_list):
    Xs, ys = [], []
    for tx in tx_list:
        sub = df[df['transaction_id'] == tx].sort_values('begin_time')
        seq = np.stack([sub[f].values.astype(np.float32) for f in FEATS], axis=1)  # (L, 6)
        # z-score per feature per sequence
        for j in range(seq.shape[1]):
            mu, sd = seq[:, j].mean(), seq[:, j].std()
            if sd > 1e-6:
                seq[:, j] = (seq[:, j] - mu) / sd
        Xs.append(seq)
        ys.append(int(tx_label[tx]))
    return Xs, np.array(ys, dtype=np.int64)

print('Building train...', flush=True)
X_tr, y_tr = build(tr_tx)
print('Building val...', flush=True)
X_va, y_va = build(va_tx)
print('Building test...', flush=True)
X_te, y_te = build(test_tx)

lens = [len(x) for x in X_tr + X_va + X_te]
print(f'Total seqs: {len(lens):,}; len min={min(lens)} p50={np.median(lens):.0f} p75={np.percentile(lens,75):.0f} p95={np.percentile(lens,95):.0f} max={max(lens)}', flush=True)

with open(f'{OUT}/seq_tensors.pkl', 'wb') as f:
    pickle.dump({'X_tr': X_tr, 'y_tr': y_tr, 'X_va': X_va, 'y_va': y_va,
                 'X_te': X_te, 'y_te': y_te, 'feats': FEATS}, f)
print(f'Saved seq_tensors.pkl in {time.time()-t0:.0f}s total', flush=True)
