#!/usr/bin/env python3
"""phase3c_attn.py — Phase 3c: TokenAttn 注意力归因 (τ=3, seed0 单模型)
（研究方案 §5.4: "最早报警信号"曲线/哪段序列/哪个特征列驱动判断）

聚合 (test owner7-8, 按 startup/run/normal 分组):
  1) CLS→seg token 注意力 profile (8 段 = 前缀时间轴) → 哪段时间窗驱动终止判断
  2) CLS→feat token 注意力 (52 列 → 按 6 通道聚合) → 哪个通道/统计量主导
输出 (docs/):
  phase3c_attn_results.json — 分组注意力 profile + top 特征
"""
import pickle, numpy as np, pandas as pd, json, os, warnings
warnings.filterwarnings('ignore')
import torch, torch.nn as nn
torch.set_num_threads(4)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, 'data'); OUT = os.path.join(BASE, 'docs')
SCHEMA = json.load(open(os.path.join(DATA, 'prefix_features_v1.json')))
FEAT_COLS = SCHEMA['feature_cols']
CH6 = {'chargingv': 0, 'charginga': 1, 'out_power': 2,
       'charging_gun_temperature1': 3, 'charging_gun_temperature2': 4, 'current_soc': 5}

# ---- 模型定义(复制自 train_prefix_tokenattn, 避免 import 触发模块级数据加载) ----
def make_len_mask(L, T, dev):
    return torch.arange(T, device=dev).unsqueeze(0) < L.unsqueeze(1).to(dev)

class BiLSTMBackbone(nn.Module):
    def __init__(self, n_seq=6, hidden=64, n_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(n_seq, hidden, num_layers=n_layers, batch_first=True,
                            bidirectional=True, dropout=0.2)
    def forward(self, x, L, maxlen):
        packed = nn.utils.rnn.pack_padded_sequence(x, L.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=maxlen)
        return out

class TokenAttnPrefix(nn.Module):
    def __init__(self, feat_dim=52, hidden=64, d=64, n_heads=4, n_layers=2, K=8, maxlen=200):
        super().__init__()
        self.backbone = BiLSTMBackbone(hidden=hidden)
        self.K = min(K, maxlen)
        self.seg_proj = nn.Linear(hidden * 2, d)
        self.feat_embed = nn.Linear(1, d)
        self.col_emb = nn.Parameter(torch.randn(feat_dim, d) * 0.01)
        self.cls = nn.Parameter(torch.randn(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, n_heads, dim_feedforward=128, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.proj = nn.Linear(d, 64)
        self.head = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, 64), nn.ReLU(),
                                  nn.Dropout(0.2), nn.Linear(64, 2))
    def segment_pool(self, A, L, maxlen):
        B, T, _ = A.shape
        mask = make_len_mask(L, T, A.device).unsqueeze(-1).float()
        Am = A * mask
        seg_len = max(1, T // self.K)
        outs = []
        for k in range(self.K):
            seg = Am[:, k*seg_len:(k+1)*seg_len, :]
            cnt = mask[:, k*seg_len:(k+1)*seg_len, :].sum(1).clamp(min=1)
            outs.append(seg.sum(1) / cnt)
        return torch.stack(outs, dim=1)
    def forward(self, x, L, f):
        maxlen = x.shape[1]
        A = self.backbone(x, L, maxlen)
        seg = self.seg_proj(self.segment_pool(A, L, maxlen))
        B = f.shape[0]
        ft = self.feat_embed(f.unsqueeze(-1)) + self.col_emb.unsqueeze(0)
        cls = self.cls.expand(B, -1, -1)
        h = self.encoder(torch.cat([cls, seg, ft], dim=1))
        return self.head(self.proj(h[:, 0]))

TAU = 3
ck = torch.load(f'{OUT}/prefix_tokenattn_tau{TAU}_s0_model.pt', map_location='cpu')
D = pickle.load(open(f'{DATA}/seq_tensors_tau{TAU}.pkl', 'rb'))

def pad(seqs):
    maxlen = max(len(s) for s in seqs)
    X = np.zeros((len(seqs), maxlen, 6), dtype=np.float32); L = np.zeros(len(seqs), dtype=np.int64)
    for i, s in enumerate(seqs):
        X[i, :len(s)] = s; L[i] = len(s)
    return X, L, maxlen

Xte, lte, MT = pad(D['X_te'])
yte = np.asarray(D['y_te']); owner_te = np.asarray(D['owner_te'])
tids_te = np.asarray(D['tids_te'])

# feat
feat_df = pd.read_parquet(f'{DATA}/prefix_feats_v1.parquet',
                          columns=FEAT_COLS + ['transaction_id', 'prefix_type', 'prefix_val', 'family'])
fmap = feat_df[(feat_df['prefix_type'] == 'time') & (feat_df['prefix_val'] == TAU)].set_index('transaction_id')
Fte = np.vstack([fmap.loc[t][FEAT_COLS].values.astype(np.float32) for t in tids_te])
fam_te = np.array([fmap.loc[t]['family'] for t in tids_te])
# 缺失填充 + z-score(训练域 stats 重算: 用全部 test? E11 严格性——归因不训练, 用 test 自身 z-score 仅影响注意力尺度
# 简化: nan→0, z-score on test(归因非训练, 不涉泄漏判断)
Fte = np.nan_to_num(Fte, nan=0.0)
mu_f = Fte.mean(0, keepdims=True); sd_f = Fte.std(0, keepdims=True)
sd_f = np.where(sd_f < 1e-8, 1.0, sd_f)
Fte = (Fte - mu_f) / sd_f

model = TokenAttnPrefix(feat_dim=52, K=8, maxlen=MT)
model.load_state_dict(ck['state']); model.eval()

# ---- 手写前向: 逐层复刻 TransformerEncoder 并抓最后一层 CLS 注意力 ----
def forward_last_attn(model, x, L, f):
    """返回最后一层 self_attn 权重 (B, T, T), 行=query"""
    maxlen = x.shape[1]
    A = model.backbone(x, L, maxlen)
    seg = model.seg_proj(model.segment_pool(A, L, maxlen))
    B = f.shape[0]
    ft = model.feat_embed(f.unsqueeze(-1)) + model.col_emb.unsqueeze(0)
    cls = model.cls.expand(B, -1, -1)
    h = torch.cat([cls, seg, ft], dim=1)              # (B, 1+8+52, d)
    attn_w = None
    for layer in model.encoder.layers:
        h2 = layer.norm1(h)
        attn_out, attn_w = layer.self_attn(h2, h2, h2, need_weights=True,
                                           average_attn_weights=True)
        h = h + attn_out
        h = h + layer._ff_block(layer.norm2(h))
    return attn_w                                     # 最后一层的 (B,T,T)

N_CLS = 1; K_SEG = 8; N_FEAT = 52
grp_of = np.where(yte == 0, 'normal', np.where(fam_te == 'startup', 'startup', 'run'))
groups = ['startup', 'run', 'normal']

cls2seg_all = {g: [] for g in groups}; cls2feat_all = {g: [] for g in groups}
B = 128
for i in range(0, len(Xte), B):
    xb, lb, fb = Xte[i:i+B], lte[i:i+B], Fte[i:i+B]
    gb = grp_of[i:i+B]
    with torch.no_grad():
        Aw = forward_last_attn(model, torch.FloatTensor(xb), torch.LongTensor(lb),
                               torch.FloatTensor(fb))
    for j in range(len(xb)):
        a = Aw[j]; g = gb[j]
        # a 形状 (T, T): 行=query; 索引0=CLS, 1..8=seg, 9..60=feat
        cls2seg_all[g].append(a[0, 1:1+K_SEG].cpu().numpy())      # CLS query → seg keys
        cls2feat_all[g].append(a[0, 1+K_SEG:1+K_SEG+N_FEAT].cpu().numpy())

# ---- 汇总: 归一化到行内 (CLS 注意力分布应 sum≈1 但只取部分列, 故用 softmax 不现实——直接平均 + 归一化) ----
def agg_norm(arrs):
    A = np.vstack([a.reshape(1, -1) for a in arrs])
    return A.mean(0)

seg_prof, feat_prof = {}, {}
for g in groups:
    if len(cls2seg_all[g]) == 0: continue
    seg_prof[g] = agg_norm(cls2seg_all[g]).tolist()
    feat_prof[g] = agg_norm(cls2feat_all[g]).tolist()

print('=== CLS→seg 注意力 (8段, 行内均值, 归一化显示) ===')
for g in groups:
    if g in seg_prof:
        v = np.asarray(seg_prof[g]); v = v / (v.sum() + 1e-12)
        print(f'  {g:<8} n={len(cls2seg_all[g]):>4} | ' + ' '.join(f'{x:.2f}' for x in v))

# feat → 按通道聚合
ch_agg = {}
for g in groups:
    if g not in feat_prof: continue
    v = np.asarray(feat_prof[g]); v = v / (v.sum() + 1e-12)
    agg = np.zeros(6)
    for cname, (col_i) in CH6.items():
        pass
    # 通道 = FEAT_COLS 前缀
    for ci, col in enumerate(FEAT_COLS):
        base = col.rsplit('_', 1)[0]
        if base in CH6:
            agg[CH6[base]] += v[ci]
    ch_agg[g] = agg.tolist()
print('\n=== CLS→feat 注意力按通道聚合 ===')
for g in groups:
    if g in ch_agg:
        print(f'  {g:<8} | ' + ' '.join(f'{x:.2f}' for x in ch_agg[g]) + '  (V/A/P/T1/T2/SOC)')

json.dump({'tau': TAU, 'seed': 0, 'seg_profile': seg_prof, 'feat_profile': feat_prof,
           'feat_by_channel': ch_agg, 'n_by_group': {g: len(cls2seg_all[g]) for g in groups}},
          open(f'{OUT}/phase3c_attn_results.json', 'w'), indent=2, ensure_ascii=False)
print(f'\n结果已存: docs/phase3c_attn_results.json')
