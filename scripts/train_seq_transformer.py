#!/usr/bin/env python3
"""序列级轻量时序 Transformer：输入整条充电序列 [L, 6]，变长用 padding + mask。
输出二分类(正常/故障)。带阈值校准。
可解释性: 1D-GradCAM(后续脚本)。
"""
import pickle, numpy as np, time, os, warnings, json
warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '4'
import torch, torch.nn as nn
torch.set_num_threads(4)
torch.manual_seed(42); np.random.seed(42)

DATA = '/Users/arthas/git/excharge/data/real/'
OUT = '/Users/arthas/git/excharge/docs/'

with open(f'{DATA}/seq_tensors.pkl', 'rb') as f:
    d = pickle.load(f)
X_tr, y_tr, X_va, y_va, X_te, y_te = d['X_tr'], d['y_tr'], d['X_va'], d['y_va'], d['X_te'], d['y_te']
FEATS = d['feats']
print(f'feats={FEATS}')
print(f'Train {len(X_tr)} (fault {y_tr.sum()}), Val {len(X_va)} (fault {y_va.sum()}), Test {len(X_te)} (fault {y_te.sum()})', flush=True)

MAXLEN = 200  # 覆盖 p95(156); 更长截断
def pad(seqs):
    B = len(seqs)
    X = np.zeros((B, MAXLEN, len(FEATS)), dtype=np.float32)
    mask = np.zeros((B, MAXLEN), dtype=bool)
    for i, s in enumerate(seqs):
        L = min(len(s), MAXLEN)
        X[i, :L] = s[:L]
        mask[i, :L] = True
    return X, mask

X_tr_p, m_tr = pad(X_tr)
X_va_p, m_va = pad(X_va)
X_te_p, m_te = pad(X_te)
print(f'Padded: tr {X_tr_p.shape}, va {X_va_p.shape}, te {X_te_p.shape}', flush=True)

class SeqTransformer(nn.Module):
    def __init__(self, n_feat=6, d_model=64, nhead=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(n_feat, d_model)
        self.pos = nn.Parameter(torch.zeros(1, MAXLEN, d_model))
        nn.init.normal_(self.pos, std=0.02)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                         dim_feedforward=128, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 2))
    def forward(self, x, mask):
        x = self.proj(x) + self.pos
        # key_padding_mask: True = ignore
        x = self.encoder(x, src_key_padding_mask=~mask)
        # mean pool over valid time steps
        pooled = (x * mask.unsqueeze(-1).float()).sum(1) / mask.unsqueeze(-1).float().sum(1).clamp(min=1)
        return self.head(pooled)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}', flush=True)
model = SeqTransformer().to(device)

pos_w = float((y_tr == 0).sum()) / max(1, int((y_tr == 1).sum()))
print(f'pos_weight={pos_w:.1f}', flush=True)
criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_w], dtype=torch.float32).to(device))
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)

X_tr_t = torch.FloatTensor(X_tr_p).to(device); m_tr_t = torch.BoolTensor(m_tr).to(device)
y_tr_t = torch.LongTensor(y_tr).to(device)
X_va_t = torch.FloatTensor(X_va_p).to(device); m_va_t = torch.BoolTensor(m_va).to(device)
y_va_t = torch.LongTensor(y_va).to(device)
X_te_t = torch.FloatTensor(X_te_p).to(device); m_te_t = torch.BoolTensor(m_te).to(device)
y_te_t = torch.LongTensor(y_te).to(device)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

BATCH = 256
EPOCHS = 40
best_f1, best_state, best_epoch = 0, None, 0
t_train = time.time()
for ep in range(EPOCHS):
    model.train()
    perm = torch.randperm(len(X_tr_t))
    tot_loss = 0
    for i in range(0, len(perm), BATCH):
        idx = perm[i:i+BATCH]
        out = model(X_tr_t[idx], m_tr_t[idx])
        loss = criterion(out, y_tr_t[idx])
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        tot_loss += loss.item()
    scheduler.step()
    model.eval()
    with torch.no_grad():
        va_out = model(X_va_t, m_va_t)
        va_pred = va_out.argmax(1).cpu().numpy()
        va_f1 = f1_score(y_va, va_pred, zero_division=0)
    if va_f1 > best_f1:
        best_f1 = va_f1; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; best_epoch = ep + 1
    if (ep+1) % 5 == 0 or ep == EPOCHS-1:
        print(f'  Epoch {ep+1}/{EPOCHS} loss={tot_loss:.4f} val_f1={va_f1:.4f} best={best_f1:.4f} (ep{best_epoch})', flush=True)
print(f'Training done in {time.time()-t_train:.0f}s', flush=True)

model.load_state_dict(best_state); model.eval()
with torch.no_grad():
    te_out = model(X_te_t, m_te_t)
    te_prob = torch.softmax(te_out, 1)[:, 1].cpu().numpy()
    te_pred = te_out.argmax(1).cpu().numpy()

res = {
    'Acc': accuracy_score(y_te, te_pred),
    'Prec': precision_score(y_te, te_pred, zero_division=0),
    'Recall': recall_score(y_te, te_pred),
    'F1': f1_score(y_te, te_pred),
    'AUC': roc_auc_score(y_te, te_prob),
}
print('\n=== SeqTransformer on Test (new owners 7-8), th=0.5 ===', flush=True)
for k, v in res.items(): print(f'  {k}: {v:.4f}', flush=True)

# PR-AUC
from sklearn.metrics import average_precision_score
pr_auc = average_precision_score(y_te, te_prob)
print(f'  PR-AUC: {pr_auc:.4f}', flush=True)
res['PR-AUC'] = pr_auc

json.dump(res, open(f'{OUT}/routeC_transformer_results.json', 'w'), indent=2)
np.save(f'{OUT}/routeC_transformer_prob.npy', te_prob)
np.save(f'{OUT}/routeC_transformer_pred.npy', te_pred)
torch.save(best_state, f'{OUT}/routeC_transformer_model.pt')
print('Saved results + model.', flush=True)
