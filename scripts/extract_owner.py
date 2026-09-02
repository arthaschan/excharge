#!/usr/bin/env python3
"""提取每个样本的 owner(站点) 标签, 供域自适应实验(实验B)使用。

从 all_data.parquet 读 transaction_id -> owner 映射, 与 fusion_data.pkl 的
train/val/test 的 tx 列表对齐, 输出 data/real/fusion_owner.pkl:
  { 'train': [owner...], 'val': [...], 'test': [...] }
owner 值: Sheet1~Sheet8 (Sheet1-6 训练源域, Sheet7-8 测试目标域)。
"""
import pandas as pd, pickle
from collections import Counter

DATA = '/Users/arthas/git/excharge/data/real/'

# 只读两列, 列裁剪加速
df = pd.read_parquet(f'{DATA}/all_data.parquet', columns=['transaction_id', 'owner'])
tx_owner = df.groupby('transaction_id')['owner'].first().to_dict()
print(f'owner mapping: {len(tx_owner):,} transactions', flush=True)
print('owner value counts:', dict(Counter(tx_owner.values())), flush=True)

with open(f'{DATA}/fusion_data.pkl', 'rb') as f:
    D = pickle.load(f)

out = {}
for split in ['train', 'val', 'test']:
    txs = D[split]['tx']
    owners = [tx_owner.get(t, 'UNK') for t in txs]
    out[split] = owners
    print(f'{split}: {len(owners)} samples, owners={dict(Counter(owners))}', flush=True)

with open(f'{DATA}/fusion_owner.pkl', 'wb') as f:
    pickle.dump(out, f)
print('Saved fusion_owner.pkl', flush=True)
