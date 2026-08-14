#!/usr/bin/env python3
"""Route B: Windowed Transformer baseline on real charging sequences.
Faithful to Nature paper: input = [chargingv, charginga, out_power] sequences
(z-scored per sequence), window >= 30 points. Pooled training, test on new owners.
"""
import pandas as pd, numpy as np, time, sys, os, warnings
warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '4'
import torch, torch.nn as nn
torch.set_num_threads(4)

DATA = '/Users/arthas/git/excharge/data/real/'
OUT  = '/Users/arthas/git/excharge/docs/'
SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)

t0 = time.time()
df = pd.read_parquet(f'{DATA}/all_data.parquet')
print(f'Loaded {len(df):,} rows in {time.time()-t0:.0f}s', flush=True)

# Sequence-level labels (transaction is homogeneous per earlier check)
tx_label = df.groupby('transaction_id')['label'].first()
tx_owner = df.groupby('transaction_id')['owner'].first()

# Only sequences >= 30 points
seq_len = df.groupby('transaction_id').size()
valid_tx = seq_len[seq_len >= 30].index
print(f'Valid transactions (>=30 pts): {len(valid_tx):,}', flush=True)

# Owner split: train owners 1-6, test owners 7-8 (new owners, like the paper)
train_owners = ['Sheet1','Sheet2','Sheet3','Sheet4','Sheet5','Sheet6']
test_owners  = ['Sheet7','Sheet8']

train_tx = [t for t in valid_tx if tx_owner[t] in train_owners]
test_tx  = [t for t in valid_tx if tx_owner[t] in test_owners]
print(f'Train tx: {len(train_tx):,} (fault {sum(tx_label[t] for t in train_tx):,})', flush=True)
print(f'Test tx:  {len(test_tx):,} (fault {sum(tx_label[t] for t in test_tx):,})', flush=True)

# Split train into train/val (stratified by label, 80/20)
labels_arr = np.array([tx_label[t] for t in train_tx])
from sklearn.model_selection import train_test_split
tr_idx, va_idx = train_test_split(np.arange(len(train_tx)), test_size=0.2, stratify=labels_arr, random_state=SEED)
tr_tx = [train_tx[i] for i in tr_idx]
va_tx = [train_tx[i] for i in va_idx]
print(f'  Train: {len(tr_tx):,} (fault {sum(tx_label[t] for t in tr_tx):,})', flush=True)
print(f'  Val:   {len(va_tx):,} (fault {sum(tx_label[t] for t in va_tx):,})', flush=True)

WINDOW = 30
STRIDE = 15
FEATS = ['chargingv', 'charginga', 'out_power']

def build_windows(tx_list):
    """Sliding windows over each sequence. Returns (X, y)."""
    Xs, ys = [], []
    for tx in tx_list:
        sub = df[df['transaction_id'] == tx].sort_values('end_time')
        v = sub['chargingv'].values.astype(np.float32)
        a = sub['charginga'].values.astype(np.float32)
        p = sub['out_power'].values.astype(np.float32)
        label = tx_label[tx]
        if len(v) < WINDOW:
            continue
        # z-score per sequence (paper eq.3)
        for arr in (v, a, p):
            mu, sd = arr.mean(), arr.std()
            if sd > 1e-6:
                arr = (arr - mu) / sd
        X = np.stack([v, a, p], axis=1)  # (L, 3)
        for s in range(0, len(X) - WINDOW + 1, STRIDE):
            Xs.append(X[s:s+WINDOW])
            ys.append(label)
    return np.array(Xs), np.array(ys)

print('Building windows (this may take a while)...', flush=True)
t1 = time.time()
X_tr, y_tr = build_windows(tr_tx)
X_va, y_va = build_windows(va_tx)
X_te, y_te = build_windows(test_tx)
print(f'Train windows: {X_tr.shape} (fault {y_tr.sum():,}) in {time.time()-t1:.0f}s', flush=True)
print(f'Val windows:   {X_va.shape} (fault {y_va.sum():,})', flush=True)
print(f'Test windows:  {X_te.shape} (fault {y_te.sum():,})', flush=True)

# Shuffle train
perm = np.random.permutation(len(X_tr))
X_tr, y_tr = X_tr[perm], y_tr[perm]

class WindowTransformer(nn.Module):
    def __init__(self, n_feat=3, d_model=64, nhead=4, n_layers=3, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(n_feat, d_model)
        self.pos = nn.Parameter(torch.zeros(1, WINDOW, d_model))
        nn.init.normal_(self.pos, std=0.02)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                         dim_feedforward=256, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 2))
    def forward(self, x):
        x = self.proj(x) + self.pos
        x = self.encoder(x)
        return self.head(x.mean(dim=1))  # mean pooling over time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}', flush=True)
model = WindowTransformer().to(device)

pos_w = float((y_tr == 0).sum()) / max(1, int((y_tr == 1).sum()))
criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_w], dtype=torch.float32).to(device))
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

X_tr_t = torch.FloatTensor(X_tr).to(device)
y_tr_t = torch.LongTensor(y_tr.astype(np.int64)).to(device)
X_va_t = torch.FloatTensor(X_va).to(device)
y_va_t = torch.LongTensor(y_va.astype(np.int64)).to(device)
X_te_t = torch.FloatTensor(X_te).to(device)
y_te_t = torch.LongTensor(y_te.astype(np.int64)).to(device)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

BATCH = 512
EPOCHS = 15
best_f1, best_state, best_epoch = 0, None, 0
t_train = time.time()
for ep in range(EPOCHS):
    model.train()
    perm = torch.randperm(len(X_tr_t))
    tot_loss = 0
    for i in range(0, len(perm), BATCH):
        idx = perm[i:i+BATCH]
        out = model(X_tr_t[idx])
        loss = criterion(out, y_tr_t[idx])
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        tot_loss += loss.item()
    scheduler.step()
    model.eval()
    with torch.no_grad():
        va_out = model(X_va_t)
        va_pred = va_out.argmax(1).cpu().numpy()
        va_f1 = f1_score(y_va, va_pred, zero_division=0)
    if va_f1 > best_f1:
        best_f1 = va_f1
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = ep + 1
    if (ep+1) % 5 == 0 or ep == EPOCHS-1:
        print(f'  Epoch {ep+1}/{EPOCHS} loss={tot_loss:.4f} val_f1={va_f1:.4f} best={best_f1:.4f} (ep{best_epoch})', flush=True)

print(f'Training done in {time.time()-t_train:.0f}s', flush=True)

# Best model on test
model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    te_out = model(X_te_t)
    te_prob = torch.softmax(te_out, 1)[:, 1].cpu().numpy()
    te_pred = te_out.argmax(1).cpu().numpy()

res = {
    'Acc': accuracy_score(y_te, te_pred),
    'Prec': precision_score(y_te, te_pred, zero_division=0),
    'Recall': recall_score(y_te, te_pred),
    'F1': f1_score(y_te, te_pred),
    'AUC': roc_auc_score(y_te, te_prob),
}
print('\n=== Windowed Transformer on Test (new owners 7-8) ===', flush=True)
for k, v in res.items():
    print(f'  {k}: {v:.4f}')
print(f'\nRecall gap vs Nature paper (73.56%): {res["Recall"]*100 - 73.56:+.2f}pp', flush=True)

# Save
import json
json.dump(res, open(f'{OUT}/routeB_transformer_results.json', 'w'), indent=2)
np.save(f'{OUT}/routeB_transformer_prob.npy', te_prob)
np.save(f'{OUT}/routeB_transformer_pred.npy', te_pred)
print('Saved results.', flush=True)
