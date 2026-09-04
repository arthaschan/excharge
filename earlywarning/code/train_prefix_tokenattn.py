#!/usr/bin/env python3
"""train_prefix_tokenattn.py — Phase 2 Token-Attn 前缀变体（结构 A Stage1 终止检测）
（gate_report_phase1 §4 / R9 探路 → 量产 / 迁移自会议版 train_c1c2.py TokenAttnFusion）

输入：data/seq_tensors_tau{τ}.pkl（序列前缀段，已 per-prefix z-score）
     + data/prefix_feats_v1.parquet（52 维表侧特征，按 tids 对齐）
模型：Token-Attn 前缀变体 = BiLSTM backbone → K 段池化 seq-token + 52 特征 numeric-token + [CLS]
      → TransformerEncoder(自注意力交互) → 二分类头（Stage1：P(终止)）
协议：加权 CE(pos_weight=neg/pos) · AdamW 1e-3 · cosine · 30 epoch · MPS/CUDA
      跨站 owner1-6 训练(内 20% val 分层) → owner7-8 test
评估：test PR-AUC(主) + AUC/F1 + 逐 owner PR-AUC（gate_phase0 §4.3：Sheet7/8 分开）

⚠️ 特征标准化：prefix_feats_v1 为原始量纲(电压数百V)，numeric-token 前必须 z-score
   （fit 仅训练域，E11），否则 attention q·k 爆炸 → softmax overflow → loss=nan。

用法：TAU=5 [SEED=42] [EPOCHS=30] [BATCH=256] python train_prefix_tokenattn.py
输出：docs/prefix_tokenattn_tau{τ}_s{seed}_results.json + _prob.npy + _model.pt(best)
"""
import pickle, numpy as np, time, os, warnings, json, sys, math
warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '4'
import torch, torch.nn as nn
import torch.nn.functional as F
torch.set_num_threads(4)

TAU = int(os.environ.get('TAU', 5))
SEED = int(os.environ.get('SEED', 42))
EPOCHS = int(os.environ.get('EPOCHS', 30))
BATCH = int(os.environ.get('BATCH', 256))
D_MODEL = int(os.environ.get('D_MODEL', 64))
N_HEADS = int(os.environ.get('N_HEADS', 4))
N_LAYERS = int(os.environ.get('N_LAYERS', 2))
FF = int(os.environ.get('FF', 128))
DROPOUT = float(os.environ.get('DROPOUT', 0.1))
K_SEG = int(os.environ.get('K_SEG', 8))
LR = float(os.environ.get('LR', 1e-3))
TAG = os.environ.get('TAG', '')          # 输出文件 tag(调参隔离,如 TAG=_lr3e-4_k4)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, 'data')
OUT = os.path.join(BASE, 'docs')
SCHEMA = json.load(open(os.path.join(DATA, 'prefix_features_v1.json')))
FEAT_COLS = SCHEMA['feature_cols']
FEAT_DIM = len(FEAT_COLS)
assert FEAT_DIM == 52

torch.manual_seed(SEED); np.random.seed(SEED)
_dev = os.environ.get('DEVICE', '')
if _dev:
    device = torch.device(_dev)
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

# ---------------- 数据载入 ----------------
import pandas as pd
t0 = time.time()
D = pickle.load(open(f'{DATA}/seq_tensors_tau{TAU}.pkl', 'rb'))
feat_df = pd.read_parquet(f'{DATA}/prefix_feats_v1.parquet',
                          columns=FEAT_COLS + ['transaction_id', 'prefix_type', 'prefix_val'])
feat_all = feat_df[(feat_df['prefix_type'] == 'time') & (feat_df['prefix_val'] == TAU)].set_index('transaction_id')

def get_feat(tids):
    return np.vstack([feat_all.loc[t][FEAT_COLS].values.astype(np.float32) for t in tids])

def pad(seqs):
    maxlen = max(len(s) for s in seqs)
    X = np.zeros((len(seqs), maxlen, 6), dtype=np.float32); L = np.zeros(len(seqs), dtype=np.int64)
    for i, s in enumerate(seqs):
        X[i, :len(s)] = s; L[i] = len(s)
    return X, L, maxlen

Xtr, ltr, MT = pad(D['X_tr']); ytr = D['y_tr']; Ftr = get_feat(D['tids_tr'])
Xva, lva, _ = pad(D['X_va']);   yva = D['y_va']; Fva = get_feat(D['tids_va'])
Xte, lte, _ = pad(D['X_te']);   yte = D['y_te']; Fte = get_feat(D['tids_te'])
# ⚠️ E11 安全缺失填充 + z-score（fit 仅训练域）：
#   power_peak_pos 在"前缀内功率全 0(尚未起充)"时为 nan(全表 1785 行,τ=1 最多 1193)。
#   LightGBM 原生处理缺失所以 Phase0/1 未暴露；深度模型 numeric-token 必须显式填充,
#   否则 mean/std 被 nan 污染 → 整列 nan → attention 溢出 → loss=nan(曾被误诊为 MPS bug)。
#   填充用训练域逐列中位数(对 val/test 同样适用,不偷看其标签)。
finite_ok = np.isfinite(Ftr)
if not finite_ok.all():
    col_med = np.where(finite_ok.any(0),
                       np.nanmedian(np.where(finite_ok, Ftr, np.nan), axis=0), 0.0)
    col_med = np.nan_to_num(col_med, nan=0.0)
    Ftr = np.where(finite_ok, Ftr, col_med)
    Fva = np.where(np.isfinite(Fva), Fva, col_med)
    Fte = np.where(np.isfinite(Fte), Fte, col_med)
    print(f'  ⚠️ 缺失填充: {int((~finite_ok).sum())} 个特征值用训练域中位数填充', flush=True)
mu_f = Ftr.mean(0, keepdims=True); sd_f = Ftr.std(0, keepdims=True)
sd_f = np.where(sd_f < 1e-8, 1.0, sd_f)
Ftr = (Ftr - mu_f) / sd_f; Fva = (Fva - mu_f) / sd_f; Fte = (Fte - mu_f) / sd_f
print(f'[τ={TAU}] train {Xtr.shape} feat{Ftr.shape} | val {Xva.shape} | test {Xte.shape} '
      f'| maxlen={MT} | device={device}', flush=True)

# ---------------- 模型（迁移自 train_c1c2 TokenAttnFusion, feat_dim=52） ----------------
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

# ---------------- 训练 + 评估（协议同 train_c1c2） ----------------
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score)

pos_w = float((ytr == 0).sum()) / max(1, int((ytr == 1).sum()))
w = torch.tensor([1.0, pos_w], dtype=torch.float32).to(device)
crit = nn.CrossEntropyLoss(weight=w)

model = TokenAttnPrefix(feat_dim=FEAT_DIM, hidden=D_MODEL, d=D_MODEL, n_heads=N_HEADS,
                        n_layers=N_LAYERS, K=K_SEG, maxlen=MT).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
nparams = sum(p.numel() for p in model.parameters())
print(f'  模型参数: {nparams:,} | pos_weight={pos_w:.1f}', flush=True)

Xtr_t = torch.FloatTensor(Xtr).to(device); ltr_t = torch.LongTensor(ltr)
Xva_t = torch.FloatTensor(Xva).to(device); lva_t = torch.LongTensor(lva)
Xte_t = torch.FloatTensor(Xte).to(device); lte_t = torch.LongTensor(lte)
Ftr_t = torch.FloatTensor(Ftr).to(device); Fva_t = torch.FloatTensor(Fva).to(device); Fte_t = torch.FloatTensor(Fte).to(device)
ytr_t = torch.LongTensor(ytr).to(device)

best_f1, best_ep, best_state = 0, 0, None
tt = time.time()
for ep in range(EPOCHS):
    model.train(); perm = torch.randperm(len(Xtr_t)); tot = 0
    for i in range(0, len(perm), BATCH):
        idx = perm[i:i+BATCH]
        if len(idx) < 2:
            continue
        out = model(Xtr_t[idx], ltr_t[idx], Ftr_t[idx])
        loss = crit(out, ytr_t[idx])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        li = loss.item()
        if not math.isfinite(li):
            print(f'  ⚠️ ep{ep+1} loss 非有限({li}) —— 终止训练便于诊断', flush=True)
            raise RuntimeError('loss diverged')
        tot += li
    sched.step(); model.eval()
    with torch.no_grad():
        vo = model(Xva_t, lva_t, Fva_t); vp = vo.argmax(1).cpu().numpy()
        vf1 = f1_score(yva, vp, zero_division=0)
    if vf1 > best_f1:
        best_f1, best_ep = vf1, ep + 1
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if (ep + 1) % 5 == 0 or ep == EPOCHS - 1:
        print(f'  ep{ep+1}/{EPOCHS} loss={tot:.3f} val_f1={vf1:.4f} best={best_f1:.4f}(ep{best_ep}) '
              f'[{time.time()-tt:.0f}s]', flush=True)

model.load_state_dict(best_state); model.eval()
with torch.no_grad():
    te = model(Xte_t, lte_t, Fte_t)
    prob = torch.softmax(te, 1)[:, 1].cpu().numpy()
    vprob = torch.softmax(model(Xva_t, lva_t, Fva_t), 1)[:, 1].cpu().numpy()
best_th, best_cf1 = 0.5, 0
for th in np.arange(0.05, 0.96, 0.01):
    f1 = f1_score(yva, (vprob >= th).astype(int), zero_division=0)
    if f1 > best_cf1:
        best_cf1, best_th = f1, float(th)
cpred = (prob >= best_th).astype(int)

owner_te = np.asarray(D['owner_te'])
res = {
    'tau': TAU, 'seed': SEED,
    'Acc': accuracy_score(yte, te.argmax(1).cpu().numpy()),
    'Prec': precision_score(yte, cpred, zero_division=0),
    'Recall': recall_score(yte, cpred),
    'F1': f1_score(yte, cpred),
    'AUC': roc_auc_score(yte, prob),
    'PR-AUC': average_precision_score(yte, prob),
    'best_epoch': best_ep, 'best_th': best_th, 'params': nparams,
    'sec': round(time.time() - tt, 1),
    'pr_auc_s7': average_precision_score(yte[owner_te == 'Sheet7'], prob[owner_te == 'Sheet7']),
    'pr_auc_s8': average_precision_score(yte[owner_te == 'Sheet8'], prob[owner_te == 'Sheet8']),
    'n_test': int(len(yte)), 'n_fault_test': int(yte.sum()),
    'device': str(device), 'feat_dim': FEAT_DIM, 'epochs': EPOCHS,
    'lr': LR, 'K_SEG': K_SEG, 'N_LAYERS': N_LAYERS, 'D_MODEL': D_MODEL,
}
os.makedirs(OUT, exist_ok=True)
sfix = f'{TAG}' if TAG else (f'_s{SEED}' if SEED != 42 else '')
json.dump({'result': res}, open(f'{OUT}/prefix_tokenattn_tau{TAU}{sfix}_results.json', 'w'), indent=2)
np.save(f'{OUT}/prefix_tokenattn_tau{TAU}{sfix}_prob.npy', prob)
torch.save({'state': best_state, 'meta': {'tau': TAU, 'seed': SEED, 'maxlen': MT,
            'best_epoch': best_ep, 'best_f1': best_f1, 'PR-AUC': res['PR-AUC'],
            'n_test': len(yte)}}, f'{OUT}/prefix_tokenattn_tau{TAU}{sfix}_model.pt')
print(f'\n=== TokenAttn 前缀变体 τ={TAU}min (seed={SEED}) ===', flush=True)
for k in ['PR-AUC', 'AUC', 'F1', 'Recall', 'Prec']:
    print(f'  {k}: {res[k]:.4f}', flush=True)
print(f'  逐 owner PR-AUC: Sheet7={res["pr_auc_s7"]:.4f} | Sheet8={res["pr_auc_s8"]:.4f}', flush=True)
print(f'  best_ep={best_ep} best_th={best_th:.2f} params={nparams:,} {res["sec"]}s', flush=True)
print('DONE', flush=True)
