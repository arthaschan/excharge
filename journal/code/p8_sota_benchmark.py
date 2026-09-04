#!/usr/bin/env python3
"""P8 / SOTA 多模型横向对比 — Phase 1（无监督重建/单类 4 模型），GPU

背景：老师指出论文只比了教科书经典模型(XGBoost/LightGBM/Bi-LSTM)，没比期刊/顶会 SOTA 时序异常检测模型。
本脚本把「已排除(推理)」升级为「已排除(实测)」。

Phase 1 模型（无监督，train 只用 normal，test 上算 PR-AUC）：
  LSTM-AE   —— 重建式锚点（动物园复盘已报 0.140，用于验证 harness 正确）
  DeepSVDD  —— ICML 2018 深度单类
  USAD      —— KDD 2020 双解码器对抗自编码
  DAGMM     —— ICLR 2018 深度自编码 + 高斯混合

协议（与全库一致）：
  数据 reproduction/data/real/fusion_data.pkl（6通道序列[L,6]，per-sequence z-score）
  切分 owner1-6 训练(13505,故障642)/owner7-8 测试(2776,故障129)，val 3377
  无监督模型仅用 train normal(y=0) 训练；指标 = test PR-AUC（故障 vs normal 排序）
  监督参照(已跑, 落盘数字)：LightGBM 0.868 / XGBoost 0.887 / Bi-LSTM 0.351 / Token-Attn 0.918

产出：journal/docs/p8_sota_benchmark.json
"""
import pickle, numpy as np, time, os, warnings, json
warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '4'
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
torch.set_num_threads(4)

from sklearn.metrics import average_precision_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BASE)
DATA = os.path.join(ROOT, 'reproduction', 'data', 'real')
OUT = os.path.join(BASE, 'docs')
os.makedirs(OUT, exist_ok=True)

MAXLEN = 200
N_CH = 6
FLAT = MAXLEN * N_CH          # 1200
EPOCHS = int(os.environ.get('EPOCHS', 30))
BATCH = 256
SEED = 42

torch.manual_seed(SEED); np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device={device} | torch {torch.__version__}', flush=True)

# ---------------- 数据 ----------------
with open(os.path.join(DATA, 'fusion_data.pkl'), 'rb') as f:
    D = pickle.load(f)

def pad(seqs):
    B = len(seqs); X = np.zeros((B, MAXLEN, N_CH), np.float32); L = np.zeros(B, np.int64)
    for i, s in enumerate(seqs):
        n = min(len(s), MAXLEN); X[i, :n] = s[:n]; L[i] = n
    return X, L

Xtr, ltr = pad(D['train']['X_tensor']); ytr = D['train']['y'].astype(np.int64)
Xva, lva = pad(D['val']['X_tensor']);   yva = D['val']['y'].astype(np.int64)
Xte, lte = pad(D['test']['X_tensor']);  yte = D['test']['y'].astype(np.int64)
# 无监督：仅 normal 训练
m_norm_tr = ytr == 0
Xn, ln = Xtr[m_norm_tr], ltr[m_norm_tr]
print(f'train {len(Xtr)}(normal {len(Xn)}, fault {int(ytr.sum())}) | val {len(Xva)} | test {len(Xte)}(fault {int(yte.sum())})', flush=True)

def to_t(x, l):
    return torch.FloatTensor(x).to(device), torch.LongTensor(l)

# 掩码：valid 时间步 [B,T,1]
def len_mask(L, T):
    return torch.arange(T, device=device).unsqueeze(0) < torch.LongTensor(L).to(device).unsqueeze(1)

def pr_auc(score, y):
    return float(average_precision_score(y, score))

results = {}
print('=' * 70, flush=True)

# ================= LSTM-AE（锚点，应≈0.140） =================
class LSTMAE(nn.Module):
    def __init__(self, n_ch=N_CH, hidden=64, n_layers=2):
        super().__init__()
        self.hidden = hidden
        self.enc = nn.LSTM(n_ch, hidden, n_layers, batch_first=True)
        self.dec = nn.LSTM(hidden, n_ch, n_layers, batch_first=True)
    def encode(self, x, L):
        packed = pack_padded_sequence(x, L.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.enc(packed)
        return h[-1]
    def forward(self, x, L, T):
        z = self.encode(x, L).unsqueeze(1).repeat(1, T, 1)
        out, _ = self.dec(z)
        return out

def run_lstmae():
    model = LSTMAE().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xn_t, ln_t = to_t(Xn, ln)
    for ep in range(EPOCHS):
        model.train(); perm = torch.randperm(len(Xn_t)); tot = 0
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i+BATCH]
            if len(idx) < 2: continue
            x, l = Xn_t[idx], ln_t[idx]
            out = model(x, l, MAXLEN)
            m = len_mask(l, MAXLEN).unsqueeze(-1).float()
            loss = (((out - x) ** 2) * m).sum() / m.sum()
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        if (ep + 1) % 10 == 0:
            print(f'  [LSTM-AE] ep{ep+1} loss={tot:.2f}', flush=True)
    model.eval()
    def score(X, L):
        Xt, Lt = to_t(X, L)
        with torch.no_grad():
            out = model(Xt, Lt, MAXLEN)
            m = len_mask(Lt, MAXLEN).unsqueeze(-1).float()
            s = (((out - Xt) ** 2) * m).sum(dim=(1, 2)) / m.sum(dim=(1, 2)).clamp(min=1)
        return s.cpu().numpy()
    return pr_auc(score(Xte, lte), yte)

results['LSTM-AE'] = run_lstmae()
print(f'[LSTM-AE] PR-AUC = {results["LSTM-AE"]:.4f} (锚点参考 0.140)', flush=True)

# ================= DeepSVDD =================
class DeepSVDD(nn.Module):
    def __init__(self, in_dim=FLAT, hiddens=(512, 128, 64)):
        super().__init__()
        layers, prev = [], in_dim
        for h in hiddens:
            layers += [nn.Linear(prev, h, bias=False), nn.ReLU()]
            prev = h
        self.enc = nn.Sequential(*layers)
    def forward(self, xf):
        return self.enc(xf)

def run_deepsvdd():
    model = DeepSVDD().to(device)
    # 初始 center = normal 编码均值（一次前向）
    Xn_t = torch.FloatTensor(Xn.reshape(len(Xn), -1)).to(device)
    with torch.no_grad():
        c = model(Xn_t).mean(0)
    c = c.detach()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(EPOCHS):
        model.train(); perm = torch.randperm(len(Xn_t)); tot = 0
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i+BATCH]
            z = model(Xn_t[idx])
            loss = ((z - c) ** 2).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        if (ep + 1) % 10 == 0:
            print(f'  [DeepSVDD] ep{ep+1} loss={tot:.2f}', flush=True)
    model.eval()
    def score(X):
        Xt = torch.FloatTensor(X.reshape(len(X), -1)).to(device)
        with torch.no_grad():
            z = model(Xt)
            return ((z - c) ** 2).sum(1).cpu().numpy()
    return pr_auc(score(Xte), yte)

results['DeepSVDD'] = run_deepsvdd()
print(f'[DeepSVDD] PR-AUC = {results["DeepSVDD"]:.4f}', flush=True)

# ================= USAD =================
class USAD(nn.Module):
    def __init__(self, in_dim=FLAT, hiddens=(512, 256, 128)):
        super().__init__()
        h1, h2, h3 = hiddens
        self.enc = nn.Sequential(nn.Linear(in_dim, h1), nn.ReLU(),
                                 nn.Linear(h1, h2), nn.ReLU(),
                                 nn.Linear(h2, h3), nn.ReLU())
        self.dec1 = nn.Sequential(nn.Linear(h3, h2), nn.ReLU(),
                                  nn.Linear(h2, h1), nn.ReLU(),
                                  nn.Linear(h1, in_dim))
        self.dec2 = nn.Sequential(nn.Linear(h3, h2), nn.ReLU(),
                                  nn.Linear(h2, h1), nn.ReLU(),
                                  nn.Linear(h1, in_dim))
    def forward(self, x):
        z = self.enc(x)
        w1 = self.dec1(z)      # 重建 1
        w2 = self.dec2(z)      # 重建 2（第二个独立解码器，从 z）
        w3 = self.dec2(self.enc(w1))  # 对抗项：D2 重构 E(w1)（把 D1 的输出再编码再解码）
        return w1, w2, w3

def run_usad():
    model = USAD().to(device)
    opt1 = torch.optim.Adam(list(model.enc.parameters()) + list(model.dec1.parameters()), lr=1e-3)
    opt2 = torch.optim.Adam(model.dec2.parameters(), lr=1e-3)
    Xn_t = torch.FloatTensor(Xn.reshape(len(Xn), -1)).to(device)
    mse = nn.MSELoss(reduction='none')
    def m(a, b): return mse(a, b).sum(1).mean()
    # Phase 1: AE 训练（enc+dec1+dec2 一起最小化重建）
    for ep in range(EPOCHS // 2):
        model.train(); perm = torch.randperm(len(Xn_t)); tot = 0
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i+BATCH]
            x = Xn_t[idx]
            w1, w2, w3 = model(x)
            loss = m(w1, x) + m(w2, x)
            opt1.zero_grad(); opt2.zero_grad()
            loss.backward(); opt1.step(); opt2.step(); tot += loss.item()
    # Phase 2: 对抗（D2 判别器 vs enc+dec1）
    for ep in range(EPOCHS // 2):
        model.train(); perm = torch.randperm(len(Xn_t)); tot = 0
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i+BATCH]
            x = Xn_t[idx]
            w1, w2, w3 = model(x)
            # 训练 D2（判别器）：D2 要 w2≈x 但 w3=D2(E(w1)) 远离 x
            loss2 = m(w2, x) - m(w3, x.detach())
            opt2.zero_grad(); loss2.backward(retain_graph=True); opt2.step()
            # 训练 enc+dec1：w1≈x，且让 w3 远离 x（误导 D2）
            w1, w2, w3 = model(x)
            loss1 = m(w1, x) - m(w3, x.detach())
            opt1.zero_grad(); loss1.backward(); opt1.step()
    model.eval()
    def score(X):
        Xt = torch.FloatTensor(X.reshape(len(X), -1)).to(device)
        with torch.no_grad():
            w1, w2, w3 = model(Xt)
            return (0.5 * mse(w1, Xt).sum(1) + 0.5 * mse(w2, Xt).sum(1)).cpu().numpy()
    return pr_auc(score(Xte), yte)

results['USAD'] = run_usad()
print(f'[USAD] PR-AUC = {results["USAD"]:.4f}', flush=True)

# ================= DAGMM =================
class DAGMM(nn.Module):
    def __init__(self, in_dim=FLAT, z_dim=10, K=4):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim, 60), nn.Tanh(),
                                 nn.Linear(60, 30), nn.Tanh(),
                                 nn.Linear(30, z_dim))
        self.dec = nn.Sequential(nn.Linear(z_dim, 30), nn.Tanh(),
                                 nn.Linear(30, 60), nn.Tanh(),
                                 nn.Linear(60, in_dim))
        self.est = nn.Sequential(nn.Linear(z_dim + 2, 10), nn.Tanh(), nn.Dropout(0.5),
                                 nn.Linear(10, K), nn.Softmax(dim=1))
        self.z_dim, self.K = z_dim, K
        self.phi = nn.Parameter(torch.zeros(K))
        self.mu = nn.Parameter(torch.zeros(K, z_dim + 2))
        self.sig = nn.Parameter(torch.ones(K, z_dim + 2))
    def compute_zr(self, x):
        z = self.enc(x)
        xr = self.dec(z)
        # 相对欧氏距离 + 余弦相似度
        rel = torch.norm(x - xr, dim=1) / (torch.norm(x, dim=1) + 1e-8)
        cos = F.cosine_similarity(x, xr, dim=1)
        return z, rel, cos
    def energy(self, z, rel, cos):
        zf = torch.cat([z, rel.unsqueeze(1), cos.unsqueeze(1)], dim=1)  # [B, z+2]
        gamma = self.est(zf)                                             # [B, K]
        phi = torch.softmax(self.phi, dim=0)
        mu = self.mu; sig = torch.exp(self.sig)
        # -log Σ φ_k N(zf|mu_k,sig_k)
        B = zf.shape[0]
        zf = zf.unsqueeze(1)                                             # [B,1,d]
        mu = mu.unsqueeze(0); sig = sig.unsqueeze(0)                     # [1,K,d]
        logp = -0.5 * torch.sum(((zf - mu) / sig) ** 2, dim=2) - 0.5 * torch.sum(torch.log(2 * np.pi * sig ** 2), dim=2)
        logp = logp + torch.log(phi + 1e-8).unsqueeze(0)
        energy = -torch.logsumexp(logp, dim=1)                           # [B]
        return energy, gamma, zf.squeeze(1)
    def forward(self, x):
        z, rel, cos = self.compute_zr(x)
        return z, rel, cos

def run_dagmm():
    model = DAGMM().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss(reduction='none')
    lam = 0.1
    Xn_t = torch.FloatTensor(Xn.reshape(len(Xn), -1)).to(device)
    for ep in range(EPOCHS):
        model.train(); perm = torch.randperm(len(Xn_t)); tot_r = tot_e = 0
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i+BATCH]
            x = Xn_t[idx]
            z, rel, cos = model(x)
            xr = model.dec(z)
            rec = mse(xr, x).sum(1).mean()
            e, _, _ = model.energy(z, rel, cos)
            loss = rec + lam * e.mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot_r += rec.item(); tot_e += e.mean().item()
        if (ep + 1) % 10 == 0:
            print(f'  [DAGMM] ep{ep+1} rec={tot_r:.1f} energy={tot_e:.1f}', flush=True)
    model.eval()
    def score(X):
        Xt = torch.FloatTensor(X.reshape(len(X), -1)).to(device)
        with torch.no_grad():
            z, rel, cos = model(Xt)
            e, _, _ = model.energy(z, rel, cos)
            return e.cpu().numpy()
    return pr_auc(score(Xte), yte)

results['DAGMM'] = run_dagmm()
print(f'[DAGMM] PR-AUC = {results["DAGMM"]:.4f}', flush=True)

# ---------------- 汇总 ----------------
summary = {
    'phase1_unsupervised': results,
    'supervised_reference': {'LightGBM': 0.868, 'XGBoost': 0.887, 'BiLSTM': 0.351, 'TokenAttn_7seed': 0.918},
    'note': ('同数据(fusion_data.pkl 6通道序列)同切分(owner1-6训/owner7-8测)同指标(test PR-AUC)。'
             '无监督模型仅用 train normal 训练。监督参照为已落盘数字。'),
}
with open(os.path.join(OUT, 'p8_sota_benchmark.json'), 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print('=' * 70, flush=True)
print('=== Phase 1 结果（test PR-AUC）===', flush=True)
for k, v in results.items():
    print(f'  {k:<12} {v:.4f}', flush=True)
print('  监督参照: LightGBM 0.868 / XGBoost 0.887 / BiLSTM 0.351 / Token-Attn 0.918', flush=True)
print('结果已存 journal/docs/p8_sota_benchmark.json', flush=True)
