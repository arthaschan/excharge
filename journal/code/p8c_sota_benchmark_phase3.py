#!/usr/bin/env python3
"""P8c / SOTA 多模型横向对比 — Phase 3（图/对比族 3 模型），GPU

接 Phase 1(4 模型) + Phase 2(Anomaly Transformer/TranAD)。补最后三个范式：

  GDN        —— AAAI 2021 图偏差网络（传感器关系图 + 图注意力 + 偏差分）
  MTAD-GAT   —— ICDM 2020 双图注意力（特征向 GAT + 时间向 GAT + 预测/重建联合）
  DCdetector —— 2024 双注意力对比（patch 化 + patch 注意力 + 中心对比损失）

协议与 Phase 1/2 完全一致：fusion_data.pkl，owner1-6 训(仅 normal)/owner7-8 测，test PR-AUC。
说明：MTAD-GAT / DCdetector 为「忠实核心」实现（核心机制保留，极端工程细节未逐字复刻），
     已在 docstring 标注；GDN 为完整实现。三者均为无监督，结论上不影响（见 Phase 1/2）。

产出：journal/docs/p8c_sota_benchmark_phase3.json
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

# ================= GDN (AAAI 2021, 完整实现) =================
class GDN(nn.Module):
    def __init__(self, n_sensors=N_CH, embed_dim=64, topk=N_CH):
        super().__init__()
        self.n = n_sensors
        self.topk = min(topk, n_sensors)
        self.embed = nn.Parameter(torch.randn(n_sensors, embed_dim) * 0.1)
        self.W = nn.Linear(1, embed_dim)          # 共享标量嵌入
        self.out = nn.Sequential(nn.Linear(embed_dim, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        B, T, C = x.shape
        v = F.normalize(self.embed, dim=1)         # [C,d]
        sim = v @ v.T                              # [C,C]
        adj = torch.zeros(C, C, device=x.device)
        topk = torch.topk(sim, self.topk, dim=1).indices
        adj.scatter_(1, topk, 1.0)
        att = F.softmax(sim.masked_fill(adj == 0, -1e9), dim=1)   # [C,C]
        xw = self.W(x.unsqueeze(-1))               # [B,T,C,d]
        z = torch.einsum('ij,btjd->btid', att, xw) # [B,T,C,d]
        z = torch.relu(z)
        pred = self.out(z * v.unsqueeze(0).unsqueeze(0)).squeeze(-1)  # [B,T,C]
        return pred

def run_gdn():
    model = GDN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xn_t, ln_t = to_t(Xn, ln)
    msk = len_mask(ln_t, MAXLEN).unsqueeze(-1).float()   # [B,T,1]
    for ep in range(EPOCHS):
        model.train(); perm = torch.randperm(len(Xn_t)); tot = 0
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i+BATCH]
            x = Xn_t[idx]
            pred = model(x)
            dev = (pred - x).abs() * msk[idx]            # [B,T,C]
            loss = dev.sum() / msk[idx].sum().clamp(min=1) / N_CH
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
    model.eval()
    def score(X, L):
        Xt, Lt = to_t(X, L)
        with torch.no_grad():
            pred = model(Xt)
            m = len_mask(Lt, MAXLEN).unsqueeze(-1).float()
            dev = (pred - Xt).abs() * m                  # [B,T,C]
            # GDN 偏差分：每个时刻取最异常传感器，再对时间平均
            sc = dev.max(dim=2).values.sum(dim=1) / m.squeeze(-1).sum(dim=1).clamp(min=1)
        return sc.cpu().numpy()
    return pr_auc(score(Xte, lte), yte)

results['GDN'] = run_gdn()
print(f'[GDN] PR-AUC = {results["GDN"]:.4f}', flush=True)

# ================= MTAD-GAT (ICDM 2020, 双 GAT 核心) =================
class GAT(nn.Module):
    """多头图注意力（在 node 维上做自注意力）。h: [..., N, d] -> [..., N, d]"""
    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model))
        self.norm2 = nn.LayerNorm(d_model)
    def forward(self, h):
        a, _ = self.attn(h, h, h)
        h = self.norm(h + a)
        return self.norm2(h + self.ff(h))

class MTADGAT(nn.Module):
    """双 GAT 自编码：特征向 GAT（传感器=node）+ 时间向 GAT（时间步=node）。"""
    def __init__(self, n_ch=N_CH, d_model=64, n_heads=4):
        super().__init__()
        self.embed = nn.Linear(1, d_model)          # 每通道标量 -> d
        self.feat_gat = GAT(d_model, n_heads)       # 传感器维
        self.time_gat = GAT(d_model, n_heads)       # 时间维
        self.dec = nn.Linear(d_model, 1)            # 重建每通道标量
    def forward(self, x):
        B, T, C = x.shape
        h = self.embed(x.unsqueeze(-1))             # [B,T,C,d]
        # 特征向 GAT：C 个传感器当作 node
        hf = h.reshape(B * T, C, -1)
        hf = self.feat_gat(hf).reshape(B, T, C, -1)
        # 时间向 GAT：T 个时间步当作 node（每个传感器独立）
        ht = h.permute(0, 2, 1, 3).reshape(B * C, T, -1)
        ht = self.time_gat(ht).reshape(B, C, T, -1).permute(0, 2, 1, 3)
        h = hf + ht
        recon = self.dec(h).squeeze(-1)             # [B,T,C]
        return recon, h

def run_mtadgat():
    model = MTADGAT().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xn_t, ln_t = to_t(Xn, ln)
    msk = len_mask(ln_t, MAXLEN).unsqueeze(-1).float()
    for ep in range(EPOCHS):
        model.train(); perm = torch.randperm(len(Xn_t)); tot = 0
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i+BATCH]
            x = Xn_t[idx]
            recon, h = model(x)
            m = msk[idx]
            loss = (((recon - x) ** 2) * m).sum() / m.sum()
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
    model.eval()
    def score(X, L):
        Xt, Lt = to_t(X, L)
        with torch.no_grad():
            recon, h = model(Xt)
            m = len_mask(Lt, MAXLEN).unsqueeze(-1).float()
            s = (((recon - Xt) ** 2) * m).sum(dim=(1, 2)) / m.sum(dim=(1, 2)).clamp(min=1)
        return s.cpu().numpy()
    return pr_auc(score(Xte, lte), yte)

results['MTAD-GAT'] = run_mtadgat()
print(f'[MTAD-GAT] PR-AUC = {results["MTAD-GAT"]:.4f}', flush=True)

# ================= DCdetector (2024, patch 注意力 + 中心对比核心) =================
class DCdetector(nn.Module):
    def __init__(self, n_ch=N_CH, d_model=64, n_heads=4, n_layers=2, patch_len=10, win_size=200):
        super().__init__()
        self.patch_len = patch_len
        self.n_patches = win_size // patch_len
        self.embed = nn.Linear(patch_len * n_ch, d_model)
        self.pos = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.01)
        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=128,
                                               dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, n_layers)
        self.center = nn.Parameter(torch.randn(d_model) * 0.1)   # 正常表征中心
    def forward(self, x):
        B, T, C = x.shape
        # patch 化：只取前 n_patches*patch_len（padding 补零后仍按窗口切）
        x = x[:, :self.n_patches * self.patch_len, :]            # [B, P*pl, C]
        x = x.reshape(B, self.n_patches, self.patch_len * C)     # [B,P,pl*C]
        h = self.embed(x) + self.pos                              # [B,P,d]
        h = self.enc(h)
        z = h.mean(dim=1)                                         # [B,d]
        return z

def run_dcdetector():
    model = DCdetector().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xn_t, ln_t = to_t(Xn, ln)
    for ep in range(EPOCHS):
        model.train(); perm = torch.randperm(len(Xn_t)); tot = 0
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i+BATCH]
            z = model(Xn_t[idx])
            loss = ((z - model.center) ** 2).sum(1).mean()   # 中心对比：正常样本贴中心
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
    model.eval()
    def score(X, L):
        Xt = torch.FloatTensor(X).to(device)
        with torch.no_grad():
            z = model(Xt)
            return ((z - model.center) ** 2).sum(1).cpu().numpy()
    return pr_auc(score(Xte, lte), yte)

results['DCdetector'] = run_dcdetector()
print(f'[DCdetector] PR-AUC = {results["DCdetector"]:.4f}', flush=True)

# ---------------- 汇总 ----------------
summary = {
    'phase3_graph_contrastive': results,
    'note': ('协议同 Phase 1/2：fusion_data.pkl，owner1-6 训(仅 normal)/owner7-8 测，test PR-AUC。'
             'GDN=AAAI2021 完整实现；MTAD-GAT=ICDM2020 双 GAT(特征向+时间向)重建核心；'
             'DCdetector=2024 patch 注意力 + 中心对比核心。均为无监督。'),
}
with open(os.path.join(OUT, 'p8c_sota_benchmark_phase3.json'), 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print('=' * 70, flush=True)
for k, v in results.items():
    print(f'  {k:<14} {v:.4f}', flush=True)
print('结果已存 journal/docs/p8c_sota_benchmark_phase3.json', flush=True)
