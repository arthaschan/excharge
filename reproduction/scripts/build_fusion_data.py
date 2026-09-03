#!/usr/bin/env python3
"""构建融合模型对齐数据集 (Fusion-aligned dataset)。

对每条有效充电序列(>=30点)在同一遍内同时产出:
  (a) 6 通道原始时序张量 [L, 6]  (z-score 归一化, padding 至 200)
  (b) 62 维手工特征 (复用 build_enhanced_features 的 enhanced_features)
按 transaction_id 对齐, 并复现 seed=42 分层抽样 (owner1-6 训练 -> 内部 80/20 切 val, owner7-8 测试),
以保证与现有纯 Bi-LSTM (routeC) 同口径可比。

输出: data/real/fusion_data.pkl
  { 'train': {X_tensor(list[L,6]), X_feat(N,62) z-scored, y, tx},
    'val':   {...}, 'test': {...},
    'feat_cols': [62], 'seq_feats': [6], 'meta': {...} }
"""
import pandas as pd, numpy as np, pickle, time, os, warnings, json
warnings.filterwarnings('ignore')

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = _ROOT + '/data/real/'
SEED = 42
np.random.seed(SEED)

# ---------- 复制 enhanced_features (来自 build_enhanced_features.py) ----------
def enhanced_features(sub):
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
    if n >= 2:
        t1_valid = t1[~np.isnan(t1)]; t2_valid = t2[~np.isnan(t2)]
        feat['t1_slope'] = (t1[-1]-t1[0])/(n-1) if len(t1_valid)>=2 else 0
        feat['t2_slope'] = (t2[-1]-t2[0])/(n-1) if len(t2_valid)>=2 else 0
        t1_diff = np.diff(t1); feat['t1_max_jump'] = np.nanmax(np.abs(t1_diff))
        t2_diff = np.diff(t2); feat['t2_max_jump'] = np.nanmax(np.abs(t2_diff))
        half = n // 2
        feat['t1_std_2nd'] = np.nanstd(t1[half:]) if n-half > 1 else 0
        feat['t2_std_2nd'] = np.nanstd(t2[half:]) if n-half > 1 else 0
        first_third = v[:max(1, n//3)]; feat['v_first_third_min'] = first_third.min()
        feat['v_sag_from_mean'] = feat['v_mean'] - first_third.min()
        last_third = a[max(0, 2*n//3):]; feat['a_last_third_max'] = last_third.max()
        feat['a_first_last_ratio'] = a[-1]/a[0] if a[0] > 0 else np.nan
        feat['p_max_jump'] = np.nanmax(np.abs(np.diff(p)))
        soc_diff = np.diff(soc)
        feat['soc_rate'] = soc_diff.mean() if len(soc_diff) > 0 else 0
        feat['soc_last_rate'] = soc_diff[-1] if len(soc_diff) > 0 else 0
    else:
        for k in ['t1_slope','t2_slope','t1_max_jump','t2_max_jump','t1_std_2nd','t2_std_2nd',
                  'v_first_third_min','v_sag_from_mean','a_last_third_max','a_first_last_ratio',
                  'p_max_jump','soc_rate','soc_last_rate']:
            feat[k] = 0
    if n >= 6:
        seg = n // 3
        v_seg1 = v[:seg]; v_seg2 = v[seg:2*seg]; v_seg3 = v[2*seg:]
        a_seg1 = a[:seg]; a_seg2 = a[seg:2*seg]; a_seg3 = a[2*seg:]
        feat['v_seg_change_1to2'] = v_seg2.mean() - v_seg1.mean()
        feat['v_seg_change_2to3'] = v_seg3.mean() - v_seg2.mean()
        feat['a_seg_change_1to2'] = a_seg2.mean() - a_seg1.mean()
        feat['a_seg_change_2to3'] = a_seg3.mean() - a_seg2.mean()
        feat['a_seg3_max'] = a_seg3.max()
        feat['v_last3_vs_first3'] = v[-3:].mean() - v[:3].mean()
        feat['a_last3_vs_first3'] = a[-3:].mean() - a[:3].mean()
        feat['p_last3_vs_first3'] = p[-3:].mean() - p[:3].mean()
    else:
        for k in ['v_seg_change_1to2','v_seg_change_2to3','a_seg_change_1to2','a_seg_change_2to3',
                  'a_seg3_max','v_last3_vs_first3','a_last3_vs_first3','p_last3_vs_first3']:
            feat[k] = 0
    feat['t1_over_40'] = int(np.nanmax(t1) > 40)
    feat['t2_over_40'] = int(np.nanmax(t2) > 40)
    feat['t1_over_45'] = int(np.nanmax(t1) > 45)
    feat['t2_over_45'] = int(np.nanmax(t2) > 45)
    return feat

# ---------- 载入 ----------
t0 = time.time()
df = pd.read_parquet(f'{DATA}/all_data.parquet')
print(f'Loaded {len(df):,} rows in {time.time()-t0:.0f}s', flush=True)

SEQ_FEATS = ['chargingv','charginga','out_power','charging_gun_temperature1','charging_gun_temperature2','current_soc']
tx_label = df.groupby('transaction_id')['label'].first()
tx_owner = df.groupby('transaction_id')['owner'].first()
bt = df.groupby('transaction_id')['types'].first()
bt_map = {3:'LFP',6:'NMC',4:'LMO',5:'LCO',7:'LP'}

seq_len = df.groupby('transaction_id').size()
valid_tx = list(seq_len[seq_len >= 30].index)
print(f'Valid transactions (>=30 pts): {len(valid_tx):,}', flush=True)

train_owners = ['Sheet1','Sheet2','Sheet3','Sheet4','Sheet5','Sheet6']
test_owners  = ['Sheet7','Sheet8']
train_tx = [t for t in valid_tx if tx_owner[t] in train_owners]
test_tx  = [t for t in valid_tx if tx_owner[t] in test_owners]
print(f'Train tx (owners1-6): {len(train_tx):,} | Test tx (owners7-8): {len(test_tx):,}', flush=True)

# 复现 seed=42 分层抽样 (与 build_seq_tensors.py 一致)
from sklearn.model_selection import train_test_split
labels_arr = np.array([tx_label[t] for t in train_tx])
tr_idx, va_idx = train_test_split(np.arange(len(train_tx)), test_size=0.2,
                                  stratify=labels_arr, random_state=SEED)
tr_tx = [train_tx[i] for i in tr_idx]
va_tx = [train_tx[i] for i in va_idx]
print(f'  Train: {len(tr_tx):,} (fault {sum(tx_label[t] for t in tr_tx):,})', flush=True)
print(f'  Val:   {len(va_tx):,} (fault {sum(tx_label[t] for t in va_tx):,})', flush=True)

# 一次性 groupby 建索引, 避免逐条过滤 O(序列数×总行数) 的灾难复杂度
groups = {tx: sub for tx, sub in df.groupby('transaction_id', sort=False)}

def build_pair(tx):
    """返回 (tensor [L,6] z-scored, feat_dict, label, owner, battery)。"""
    sub = groups[tx].sort_values('begin_time')
    seq = np.stack([sub[f].values.astype(np.float32) for f in SEQ_FEATS], axis=1)  # (L,6)
    for j in range(seq.shape[1]):
        mu, sd = seq[:, j].mean(), seq[:, j].std()
        if sd > 1e-6:
            seq[:, j] = (seq[:, j] - mu) / sd
    fdict = enhanced_features(sub)
    return seq, fdict, int(tx_label[tx]), tx_owner[tx], bt_map.get(bt.get(tx, 0), 'UNK')

def assemble(tx_list):
    Xt, Xf, y, txs = [], [], [], []
    for tx in tx_list:
        seq, fdict, lab, own, btype = build_pair(tx)
        if not fdict:
            continue
        Xt.append(seq)
        fdict['bt_'+btype] = 1  # 单热(只命中对应电池类型)
        Xf.append(fdict)
        y.append(lab)
        txs.append(tx)
    return Xt, Xf, np.array(y, dtype=np.int64), txs

print('Building train...', flush=True); tr_Xt, tr_Xf, tr_y, tr_txs = assemble(tr_tx)
print('Building val...', flush=True);   va_Xt, va_Xf, va_y, va_txs = assemble(va_tx)
print('Building test...', flush=True);   te_Xt, te_Xf, te_y, te_txs = assemble(test_tx)

# 统一特征列 (以训练集出现的列为准, 并固定电池类型列)
all_bt = ['bt_LFP','bt_NMC','bt_LMO','bt_LCO','bt_LP']
base_cols = [c for c in tr_Xf[0].keys() if not c.startswith('bt_')]
feat_cols = base_cols + all_bt

def to_matrix(Xf_list):
    M = np.zeros((len(Xf_list), len(feat_cols)), dtype=np.float32)
    for i, fd in enumerate(Xf_list):
        for j, c in enumerate(feat_cols):
            M[i, j] = fd.get(c, 0.0)
    return M

Xtr_f = to_matrix(tr_Xf); Xva_f = to_matrix(va_Xf); Xte_f = to_matrix(te_Xf)
# 缺失/NaN 用训练集中位数填充
med = np.nanmedian(Xtr_f, axis=0)
Xtr_f = np.where(np.isnan(Xtr_f), med, Xtr_f)
Xva_f = np.where(np.isnan(Xva_f), med, Xva_f)
Xte_f = np.where(np.isnan(Xte_f), med, Xte_f)
# z-score 归一化 (fit on train) 供 NN 使用
mu = Xtr_f.mean(axis=0); sd = Xtr_f.std(axis=0)
sd = np.where(sd < 1e-8, 1.0, sd)
Xtr_f = (Xtr_f - mu) / sd
Xva_f = (Xva_f - mu) / sd
Xte_f = (Xte_f - mu) / sd

print(f'Feature matrix: train {Xtr_f.shape}, val {Xva_f.shape}, test {Xte_f.shape}', flush=True)
print(f'Feat cols ({len(feat_cols)}): {feat_cols}', flush=True)

out = {
    'train': {'X_tensor': tr_Xt, 'X_feat': Xtr_f, 'y': tr_y, 'tx': tr_txs},
    'val':   {'X_tensor': va_Xt, 'X_feat': Xva_f, 'y': va_y, 'tx': va_txs},
    'test':  {'X_tensor': te_Xt, 'X_feat': Xte_f, 'y': te_y, 'tx': te_txs},
    'feat_cols': feat_cols, 'seq_feats': SEQ_FEATS,
    'meta': {'n_train': len(tr_y), 'n_val': len(va_y), 'n_test': len(te_y),
             'fault_train': int(tr_y.sum()), 'fault_val': int(va_y.sum()), 'fault_test': int(te_y.sum()),
             'seed': SEED, 'n_features': len(feat_cols)}
}
with open(f'{DATA}/fusion_data.pkl', 'wb') as f:
    pickle.dump(out, f)
print(f'Saved fusion_data.pkl in {time.time()-t0:.0f}s total', flush=True)
print('Summary:', out['meta'])
