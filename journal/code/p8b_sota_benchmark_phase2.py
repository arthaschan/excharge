#!/usr/bin/env python3
"""P8b / SOTA 多模型横向对比 — Phase 2（transformer 级无监督 SOTA），GPU

接 p8_sota_benchmark.py Phase 1（LSTM-AE 0.197 / DeepSVDD 0.195 / USAD 0.029 / DAGMM 0.032）。
本脚本补两个最常被审稿人问到的 transformer 级 SOTA：

  Anomaly Transformer —— ICLR 2022（关联差异 association discrepancy）
  TranAD —— VLDB 2022（transformer 自编码 + 对抗，本实现取重建核心，未实现完整 meta-learning，
           已在 docstring 标注，作为「transformer 重建式」代表）

协议与 Phase 1 完全一致：fusion_data.pkl，owner1-6 训(仅 normal)/owner7-8 测，test PR-AUC。

产出：journal/docs/p8b_sota_benchmark_phase2.json
"""
import pickle, numpy as np, time, os, warnings, json, math
warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '4'
import torch, torch.nn as nn
import torch.nn.functional as F
torch.set_num_threads(4)
from sklearn.metrics import average_precision_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BASE)
DATA = os.path.join(ROOT, 'reproduction', 'data', 'real')
OUT = os.path.join(BASE, 'docs')
os.makedirs(OUT, exist_ok=True)

MAXLEN = 200
N_CH = 6
EPOCHS = int(os.environ.get('EPOCHS', 30))
BATCH = 256
SEED = 42
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 3

torch.manual_seed(SEED); np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device={device}', flush=True)

with open(os.path.join(DATA, 'fusion_data.pkl'), 'rb') as f:
    D = pickle.load(f)

def pad(seqs):
    B = len(seqs); X = np.zeros((B, MAXLEN, N_CH), np.float32); L = np.zeros(B, np.int64)
    for i, s in enumerate(seqs):
        n = min(len(s), MAXLEN); X[i, :n] = s[:n]; L[i] = n
    return X, L

Xtr, ltr = pad(D['train']['X_tensor']); ytr = D['train']['y'].astype(np.int64)
Xte, lte = pad(D['test']['X_tensor']);  yte = D['test']['y'].astype(np.int64)
Xn, ln = Xtr[ytr == 0], ltr[ytr == 0]
print(f'train {len(Xtr)}(normal {len(Xn)}) | test {len(Xte)}(fault {int(yte.sum())})', flush=True)

def to_t(x, l):
    return torch.FloatTensor(x).to(device), torch.LongTensor(l)

def len_mask(L, T):
    return torch.arange(T, device=device).unsqueeze(0) < torch.LongTensor(L).to(device).unsqueeze(1)

def pr_auc(s, y): return float(average_precision_score(y, s))

results = {}

# ================= Anomaly Transformer (ICLR 2022) =================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.shape[1]]

class AnomalyAttention(nn.Module):
    def __init__(self, d_model, n_heads, win_size):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.win_size = win_size
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        self.sigma = nn.Parameter(torch.ones(1))   # 可学习高斯带宽
    def forward(self, x):
        B, L, _ = x.shape
        H = self.n_heads
        Q = self.Wq(x).view(B, L, H, self.d_k).transpose(1, 2)   # [B,H,L,dk]
        K = self.Wk(x).view(B, L, H, self.d_k).transpose(1, 2)
        V = self.Wv(x).view(B, L, H, self.d_k).transpose(1, 2)
        series = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # [B,H,L,L]
        # prior association（高斯，按行归一化）
        idx = torch.arange(L, device=x.device).float()
        dist2 = (idx.unsqueeze(0) - idx.unsqueeze(1)) ** 2
        sigma = torch.abs(self.sigma) + 1e-6
        prior = torch.exp(-dist2 / (2 * sigma * sigma))       # [L,L]
        prior = prior / (prior.sum(-1, keepdim=True) + 1e-8)
        prior = prior.unsqueeze(0).unsqueeze(0)               # [1,1,L,L]
        series_soft = torch.softmax(series, dim=-1)
        out = torch.matmul(series_soft, V)                    # [B,H,L,dk]
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        out = self.Wo(out)
        return out, series_soft, prior

class AnomalyTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, win_size, dropout=0.1):
        super().__init__()
        self.attn = AnomalyAttention(d_model, n_heads, win_size)
        self.ff = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(d_model * 4, d_model))
        self.norm1 = nn.LayerNorm(d_model); self.norm2 = nn.LayerNorm(d_model)
    def forward(self, x):
        a, series, prior = self.attn(x)
        x = self.norm1(x + a)
        x = self.norm2(x + self.ff(x))
        return x, series, prior

class AnomalyTransformer(nn.Module):
    def __init__(self, n_ch=N_CH, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, win_size=200):
        super().__init__()
        self.embed = nn.Linear(n_ch, d_model)
        self.pos = PositionalEncoding(d_model, win_size)
        self.blocks = nn.ModuleList([AnomalyTransformerBlock(d_model, n_heads, win_size) for _ in range(n_layers)])
        self.out = nn.Linear(d_model, n_ch)
    def forward(self, x):
        h = self.pos(self.embed(x))
        series_list, prior_list = [], []
        for b in self.blocks:
            h, series, prior = b(h)
            series_list.append(series); prior_list.append(prior)
        recon = self.out(h)
        return recon, series_list, prior_list

def ass_discrepancy(series_list, prior_list):
    """对称 KL 的逐点关联差异，跨层跨头取均值 → [B,L]"""
    eps = 1e-8
    tot = None
    for series, prior in zip(series_list, prior_list):
        kl_ps = (prior * (torch.log(prior + eps) - torch.log(series + eps))).sum(-1)  # [B,H,L]
        kl_sp = (series * (torch.log(series + eps) - torch.log(prior + eps))).sum(-1)
        ad = (kl_ps + kl_sp).mean(1)   # [B,L]
        tot = ad if tot is None else tot + ad
    return tot / len(series_list)

def run_anomaly_transformer():
    model = AnomalyTransformer().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xn_t, ln_t = to_t(Xn, ln)
    msk = len_mask(ln_t, MAXLEN).unsqueeze(-1).float()  # [B,L,1]
    for ep in range(EPOCHS):
        model.train(); perm = torch.randperm(len(Xn_t)); tot = 0
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i+BATCH]
            x = Xn_t[idx]
            recon, s_list, p_list = model(x)
            m = msk[idx]
            rec = (((recon - x) ** 2) * m).sum() / m.sum()
            ad = ass_discrepancy(s_list, p_list)        # [B,L]
            ad = (ad * m.squeeze(-1)).sum() / m.squeeze(-1).sum()
            loss = rec - 0.5 * ad    # 最小化重建，最小化关联差异（正常点关联差异应小）
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
    model.eval()
    def score(X, L):
        Xt, Lt = to_t(X, L)
        with torch.no_grad():
            recon, s_list, p_list = model(Xt)
            m = len_mask(Lt, MAXLEN).unsqueeze(-1).float()
            rec = ((recon - Xt) ** 2) * m          # [B,L,C]
            rec_t = rec.sum(-1)                     # [B,L]
            ad = ass_discrepancy(s_list, p_list)    # [B,L]
            w = torch.softmax(-ad, dim=-1)          # 关联差异小→权重高（正常点）
            sc = (rec_t * w).sum(-1) / m.squeeze(-1).sum(-1).clamp(min=1)
        return sc.cpu().numpy()
    return pr_auc(score(Xte, lte), yte)

results['AnomalyTransformer'] = run_anomaly_transformer()
print(f'[AnomalyTransformer] PR-AUC = {results["AnomalyTransformer"]:.4f}', flush=True)

# ================= TranAD (VLDB 2022, transformer AE 重建核心) =================
class TranADAE(nn.Module):
    """Transformer 自编码重建（TranAD 的重建核心，未实现完整 meta-learning 阶段）"""
    def __init__(self, n_ch=N_CH, d_model=D_MODEL, n_heads=N_HEADS, n_layers=2, win_size=200):
        super().__init__()
        self.embed = nn.Linear(n_ch, d_model)
        self.pos = PositionalEncoding(d_model, win_size)
        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=128,
                                               dropout=0.1, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, n_layers)
        self.dec = nn.Linear(d_model, n_ch)
    def forward(self, x):
        h = self.pos(self.embed(x))
        h = self.enc(h)
        return self.dec(h)

def run_tranad():
    model = TranADAE().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xn_t, ln_t = to_t(Xn, ln)
    msk = len_mask(ln_t, MAXLEN).unsqueeze(-1).float()
    for ep in range(EPOCHS):
        model.train(); perm = torch.randperm(len(Xn_t)); tot = 0
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i+BATCH]
            x = Xn_t[idx]
            recon = model(x)
            m = msk[idx]
            loss = (((recon - x) ** 2) * m).sum() / m.sum()
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
    model.eval()
    def score(X, L):
        Xt, Lt = to_t(X, L)
        with torch.no_grad():
            recon = model(Xt)
            m = len_mask(Lt, MAXLEN).unsqueeze(-1).float()
            s = (((recon - Xt) ** 2) * m).sum(dim=(1, 2)) / m.sum(dim=(1, 2)).clamp(min=1)
        return s.cpu().numpy()
    return pr_auc(score(Xte, lte), yte)

results['TranAD'] = run_tranad()
print(f'[TranAD] PR-AUC = {results["TranAD"]:.4f}', flush=True)

# ---------------- 汇总 ----------------
summary = {
    'phase2_transformer': results,
    'note': ('协议同 Phase 1：fusion_data.pkl，owner1-6 训(仅 normal)/owner7-8 测，test PR-AUC。'
             'AnomalyTransformer=ICLR 2022 关联差异(完整实现)；TranAD=VLDB 2022 的 transformer 重建核心'
             '(未实现 meta-learning 阶段，作 transformer 重建式代表)。'),
}
with open(os.path.join(OUT, 'p8b_sota_benchmark_phase2.json'), 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print('=' * 70, flush=True)
for k, v in results.items():
    print(f'  {k:<20} {v:.4f}', flush=True)
print('结果已存 journal/docs/p8b_sota_benchmark_phase2.json', flush=True)
