#!/usr/bin/env python3
"""P9 / 对照实验 — 无监督重建式模型「给标签微调」后是否追上监督路线，GPU

问题（回应老师 + 坐实「监督是主场」）：
  无监督 SOTA 全失败，根因到底「没标签」还是「范式错（吃原始序列重建）」？
  本实验把**标签给重建式模型**（在其编码器上加分类头、用标签监督训练），
  若给标签后能追上 GBDT(0.868) → 根因是「没标签」；若仍远低于 GBDT → 根因是「范式错」。

做法：3 个重建式模型的编码器 + 分类头，用**全部训练集(含故障标签)** 监督训练（加权 CE），
      对比其无监督版 PR-AUC 与监督 GBDT 参照。

  LSTM-AE 编码器(BiLSTM) → 分类头     （对应纯 Bi-LSTM 监督，已知 ≈0.351）
  DeepSVDD 编码器(扁平 MLP) → 分类头
  AnomalyTransformer 编码器 → 分类头

产出：journal/docs/p9_supervised_finetune.json
"""
import pickle, numpy as np, time, os, warnings, json, math
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
FLAT = MAXLEN * N_CH
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
print(f'train {len(Xtr)}(fault {int(ytr.sum())}) | test {len(Xte)}(fault {int(yte.sum())})', flush=True)

Xtr_t = torch.FloatTensor(Xtr).to(device); ltr_t = torch.LongTensor(ltr)
Xte_t = torch.FloatTensor(Xte).to(device); lte_t = torch.LongTensor(lte)
ytr_t = torch.LongTensor(ytr).to(device)

pos_w = float((ytr == 0).sum()) / max(1, int((ytr == 1).sum()))
crit = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_w], device=device))

def len_mask(L, T):
    return torch.arange(T, device=device).unsqueeze(0) < torch.LongTensor(L).to(device).unsqueeze(1)

def pr_auc(s, y): return float(average_precision_score(y, s))

def train_supervised(model, make_input, epochs=EPOCHS):
    """make_input(idx, X, L) -> model input tuple（对 train 与 test 复用）"""
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    N = len(Xtr_t)
    for ep in range(epochs):
        model.train(); perm = torch.randperm(N); tot = 0
        for i in range(0, N, BATCH):
            idx = perm[i:i+BATCH]
            if len(idx) < 2: continue
            out = model(*make_input(idx, Xtr_t, ltr_t))
            loss = crit(out, ytr_t[idx])
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        if (ep + 1) % 10 == 0:
            print(f'    ep{ep+1} loss={tot:.2f}', flush=True)
    model.eval()
    with torch.no_grad():
        out = model(*make_input(torch.arange(len(Xte_t)), Xte_t, lte_t))
        prob = torch.softmax(out, 1)[:, 1].cpu().numpy()
    return prob

results = {}

# ---------- 1. BiLSTM 编码器（LSTM-AE 的编码器）+ 分类头 ----------
class SupBiLSTM(nn.Module):
    def __init__(self, n_ch=N_CH, hidden=64, n_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(n_ch, hidden, n_layers, batch_first=True, bidirectional=True)
        self.head = nn.Linear(hidden * 2, 2)
    def forward(self, x, L):
        packed = pack_padded_sequence(x, L.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        z = torch.cat([h[-2], h[-1]], dim=1)
        return self.head(z)

m = SupBiLSTM().to(device)
prob = train_supervised(m, lambda idx, X, L: (X[idx], L[idx]))
results['BiLSTM-encoder(supervised)'] = pr_auc(prob, yte)
print(f'[1/3] BiLSTM 编码器 + 分类头(监督) PR-AUC = {results["BiLSTM-encoder(supervised)"]:.4f}', flush=True)

# ---------- 2. DeepSVDD 编码器（扁平 MLP）+ 分类头 ----------
class SupMLP(nn.Module):
    def __init__(self, in_dim=FLAT, hiddens=(512, 128, 64)):
        super().__init__()
        layers, prev = [], in_dim
        for h in hiddens:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.enc = nn.Sequential(*layers)
        self.head = nn.Linear(64, 2)
    def forward(self, xf):
        return self.head(self.enc(xf))

m = SupMLP().to(device)
prob = train_supervised(m, lambda idx, X, L: (X[idx].reshape(len(idx), -1),))
results['DeepSVDD-encoder(supervised)'] = pr_auc(prob, yte)
print(f'[2/3] DeepSVDD 编码器(扁平 MLP) + 分类头(监督) PR-AUC = {results["DeepSVDD-encoder(supervised)"]:.4f}', flush=True)

# ---------- 3. Anomaly Transformer 编码器 + 分类头 ----------
class SupAnoTrans(nn.Module):
    def __init__(self, n_ch=N_CH, d_model=64, n_heads=4, n_layers=3, win_size=200):
        super().__init__()
        self.embed = nn.Linear(n_ch, d_model)
        pe = torch.zeros(win_size, d_model)
        pos = torch.arange(win_size).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
        self.register_buffer('pe', pe.unsqueeze(0))
        enc_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=128, dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, n_layers)
        self.head = nn.Linear(d_model, 2)
    def forward(self, x, L):
        h = self.embed(x) + self.pe[:, :x.shape[1]]
        h = self.enc(h)
        m = len_mask(L, x.shape[1]).unsqueeze(-1).float()
        z = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(z)

m = SupAnoTrans().to(device)
prob = train_supervised(m, lambda idx, X, L: (X[idx], L[idx]))
results['AnomalyTransformer-encoder(supervised)'] = pr_auc(prob, yte)
print(f'[3/3] AnomalyTransformer 编码器 + 分类头(监督) PR-AUC = {results["AnomalyTransformer-encoder(supervised)"]:.4f}', flush=True)

# ---------- 汇总（无监督 vs 给标签 vs 监督 GBDT） ----------
summary = {
    'supervised_finetune': results,
    'unsupervised_reference': {'LSTM-AE': 0.1967, 'DeepSVDD': 0.1947, 'AnomalyTransformer': 0.0665},
    'supervised_gbdt_reference': {'LightGBM': 0.868, 'XGBoost': 0.887, 'TokenAttn_7seed': 0.918},
    'note': ('把标签给重建式模型的编码器(加分类头监督训练)，对比其无监督版与监督 GBDT。'
             '若给标签后仍远低于 GBDT → 根因是「范式错(吃原始序列)」而非「没标签」。'
             '协议：owner1-6 全量(含故障)监督训练 / owner7-8 测，加权 CE，PR-AUC。'),
}
with open(os.path.join(OUT, 'p9_supervised_finetune.json'), 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print('=' * 70, flush=True)
print('=== P9 对照：无监督 → 给标签(监督) → GBDT ===', flush=True)
for name, unsup, sup in [('LSTM-AE', 0.1967, results['BiLSTM-encoder(supervised)']),
                          ('DeepSVDD', 0.1947, results['DeepSVDD-encoder(supervised)']),
                          ('AnomalyTransformer', 0.0665, results['AnomalyTransformer-encoder(supervised)'])]:
    print(f'  {name:<20} 无监督 {unsup:.4f} → 给标签 {sup:.4f} (GBDT 参照 {summary["supervised_gbdt_reference"]["LightGBM"]:.3f})', flush=True)
print('结果已存 journal/docs/p9_supervised_finetune.json', flush=True)
