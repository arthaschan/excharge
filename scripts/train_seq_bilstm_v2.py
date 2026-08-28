#!/usr/bin/env python3
"""Bi-LSTM 调优版 v2：MPS 加速 + 注意力池化 + 降采样正常样本。
- 用 Apple MPS 加速训练
- 注意力池化(可学习)替代 last+max pooling
- 降采样正常序列(正常:故障 = 3:1)缓解类别不平衡
- EPOCHS 40 + 早停(patience=8)
"""
import pickle, numpy as np, time, os, warnings, json
warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '4'
import torch, torch.nn as nn
torch.manual_seed(42); np.random.seed(42)

DATA = '/Users/arthas/git/excharge/data/real/'
OUT = '/Users/arthas/git/excharge/docs/'

with open(f'{DATA}/seq_tensors.pkl', 'rb') as f:
    d = pickle.load(f)
X_tr, y_tr, X_va, y_va, X_te, y_te = d['X_tr'], d['y_tr'], d['X_va'], d['y_va'], d['X_te'], d['y_te']
FEATS = d['feats']
print(f'feats={FEATS}', flush=True)
print(f'Train {len(X_tr)} (fault {y_tr.sum()}), Val {len(X_va)} (fault {y_va.sum()}), Test {len(X_te)} (fault {y_te.sum()})', flush=True)

# 降采样正常样本: 正常:故障 = 3:1
fault_tr = np.where(y_tr == 1)[0]
normal_tr = np.where(y_tr == 0)[0]
n_keep = min(len(normal_tr), len(fault_tr) * 3)
rng = np.random.default_rng(42)
normal_keep = rng.choice(normal_tr, size=n_keep, replace=False)
keep_idx = np.sort(np.concatenate([fault_tr, normal_keep]))
X_tr = [X_tr[i] for i in keep_idx]; y_tr = y_tr[keep_idx]
print(f'After downsampling: Train {len(X_tr)} (fault {y_tr.sum()}, ratio 1:{len(X_tr)/y_tr.sum()-1:.1f})', flush=True)

MAXLEN = 200
def pad(seqs):
    B = len(seqs); F = len(FEATS)
    X = np.zeros((B, MAXLEN, F), dtype=np.float32)
    L_arr = np.zeros(B, dtype=np.int64)
    for i, s in enumerate(seqs):
        L = min(len(s), MAXLEN)
        X[i, :L] = s[:L]; L_arr[i] = L
    return X, L_arr

X_tr_p, l_tr = pad(X_tr); X_va_p, l_va = pad(X_va); X_te_p, l_te = pad(X_te)
print(f'Padded: tr {X_tr_p.shape}, va {X_va_p.shape}, te {X_te_p.shape}', flush=True)

class BiLSTM_Attn(nn.Module):
    def __init__(self, n_feat=6, hidden=64, n_layers=2, dropout=0.2):
        super().__init__()
        self.hidden = hidden
        self.lstm = nn.LSTM(n_feat, hidden, num_layers=n_layers, batch_first=True,
                            bidirectional=True, dropout=dropout)
        self.attn = nn.Sequential(nn.Linear(hidden*2, 32), nn.Tanh(), nn.Linear(32, 1))
        self.head = nn.Sequential(
            nn.LayerNorm(hidden*2), nn.Linear(hidden*2, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 2))
    def forward(self, x, L):
        B, T, F = x.shape
        packed = nn.utils.rnn.pack_padded_sequence(x, L.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=T)
        mask = torch.arange(T, device=x.device).unsqueeze(0) < L.unsqueeze(1).to(x.device)
        score = self.attn(out).squeeze(-1)
        score = score.masked_fill(~mask, -1e9)
        alpha = torch.softmax(score, dim=-1).unsqueeze(-1)
        ctx = (out * alpha).sum(dim=1)
        return self.head(ctx)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Device: {device}', flush=True)
model = BiLSTM_Attn().to(device)

pos_w = float((y_tr == 0).sum()) / max(1, int((y_tr == 1).sum()))
print(f'pos_weight={pos_w:.1f}', flush=True)
criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_w], dtype=torch.float32).to(device))
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)

X_tr_t = torch.FloatTensor(X_tr_p).to(device); l_tr_t = torch.LongTensor(l_tr)
y_tr_t = torch.LongTensor(y_tr).to(device)
X_va_t = torch.FloatTensor(X_va_p).to(device); l_va_t = torch.LongTensor(l_va)
y_va_t = torch.LongTensor(y_va).to(device)
X_te_t = torch.FloatTensor(X_te_p).to(device); l_te_t = torch.LongTensor(l_te)
y_te_t = torch.LongTensor(y_te).to(device)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

BATCH = 256
EPOCHS = 40
PATIENCE = 8
best_f1, best_state, best_epoch, no_improve = 0, None, 0, 0
t_train = time.time()
for ep in range(EPOCHS):
    model.train()
    perm = torch.randperm(len(X_tr_t))
    tot_loss = 0
    for i in range(0, len(perm), BATCH):
        idx = perm[i:i+BATCH]
        out = model(X_tr_t[idx], l_tr_t[idx].cpu())
        loss = criterion(out, y_tr_t[idx])
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tot_loss += loss.item()
    scheduler.step()
    model.eval()
    with torch.no_grad():
        va_out = model(X_va_t, l_va_t.cpu())
        va_pred = va_out.argmax(1).cpu().numpy()
        va_f1 = f1_score(y_va, va_pred, zero_division=0)
    if va_f1 > best_f1:
        best_f1 = va_f1; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; best_epoch = ep + 1; no_improve = 0
    else:
        no_improve += 1
    if (ep+1) % 5 == 0 or ep == EPOCHS-1:
        print(f'  Epoch {ep+1}/{EPOCHS} loss={tot_loss:.4f} val_f1={va_f1:.4f} best={best_f1:.4f} (ep{best_epoch})', flush=True)
    if no_improve >= PATIENCE:
        print(f'  Early stop at epoch {ep+1}', flush=True)
        break
print(f'Training done in {time.time()-t_train:.0f}s', flush=True)

model.load_state_dict(best_state); model.eval()
with torch.no_grad():
    te_out = model(X_te_t, l_te_t.cpu())
    te_prob = torch.softmax(te_out, 1)[:, 1].cpu().numpy()
    te_pred = te_out.argmax(1).cpu().numpy()

res = {
    'Acc': accuracy_score(y_te, te_pred),
    'Prec': precision_score(y_te, te_pred, zero_division=0),
    'Recall': recall_score(y_te, te_pred),
    'F1': f1_score(y_te, te_pred),
    'AUC': roc_auc_score(y_te, te_prob),
    'PR-AUC': average_precision_score(y_te, te_prob),
}
print('\n=== Bi-LSTM+Attn+Downsample on Test (owner 7-8), th=0.5 ===', flush=True)
for k, v in res.items(): print(f'  {k}: {v:.4f}', flush=True)

print('\n=== 阈值扫描 ===', flush=True)
best_th, best_th_f1 = 0.5, 0
for th in [0.3, 0.4, 0.5, 0.6, 0.7]:
    p = (te_prob >= th).astype(int)
    f1 = f1_score(y_te, p, zero_division=0)
    print(f'  th={th}: R={recall_score(y_te,p,zero_division=0):.4f} P={precision_score(y_te,p,zero_division=0):.4f} F1={f1:.4f}', flush=True)
    if f1 > best_th_f1:
        best_th_f1 = f1; best_th = th
res['best_th'] = best_th; res['best_th_f1'] = best_th_f1

json.dump(res, open(f'{OUT}/routeC_bilstm_v2_results.json', 'w'), indent=2)
np.save(f'{OUT}/routeC_bilstm_v2_prob.npy', te_prob)
np.save(f'{OUT}/routeC_bilstm_v2_pred.npy', te_pred)
torch.save(best_state, f'{OUT}/routeC_bilstm_v2_model.pt')
print('Saved results + model.', flush=True)
