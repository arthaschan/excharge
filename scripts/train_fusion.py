#!/usr/bin/env python3
"""融合模型: Bi-LSTM 128维序列表示 + 62维手工特征拼接 -> 分类头。
支持断点续训 (RESUME=1): 从 fusion_ckpt.pt 恢复, 每次跑一小段以规避前台命令 ~10min 上限。
训练 owner1-6 (val 早停), 测试 owner7-8。checkpoint 每轮落盘。
"""
import pickle, numpy as np, time, os, warnings, json
warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '4'
import torch, torch.nn as nn
torch.set_num_threads(4); torch.manual_seed(42); np.random.seed(42)

DATA = '/Users/arthas/git/excharge/data/real/'
OUT  = '/Users/arthas/git/excharge/docs/'
CKPT = f'{OUT}/fusion_ckpt.pt'
RESUME = os.environ.get('RESUME', '0') == '1'
EPOCHS = int(os.environ.get('EPOCHS', 30))

with open(f'{DATA}/fusion_data.pkl', 'rb') as f:
    D = pickle.load(f)
FEAT_DIM = D['meta']['n_features']
SEQ_FEATS = D['seq_feats']; N_SEQ = len(SEQ_FEATS); MAXLEN = 200

def pad(seqs):
    B = len(seqs); X = np.zeros((B, MAXLEN, N_SEQ), dtype=np.float32); L = np.zeros(B, dtype=np.int64)
    for i, s in enumerate(seqs):
        n = min(len(s), MAXLEN); X[i, :n] = s[:n]; L[i] = n
    return X, L

Xtr, ltr = pad(D['train']['X_tensor']); ytr = D['train']['y']
Xva, lva = pad(D['val']['X_tensor']);   yva = D['val']['y']
Xte, lte = pad(D['test']['X_tensor']);  yte = D['test']['y']
Ftr = D['train']['X_feat'].astype(np.float32); Fva = D['val']['X_feat'].astype(np.float32); Fte = D['test']['X_feat'].astype(np.float32)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print('Device:', device, flush=True)

class FusionModel(nn.Module):
    def __init__(self, n_seq=6, hidden=64, n_layers=2, feat_dim=62, dropout=0.2):
        super().__init__()
        self.hidden = hidden
        self.lstm = nn.LSTM(n_seq, hidden, num_layers=n_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.head = nn.Sequential(nn.LayerNorm(hidden*2 + feat_dim), nn.Linear(hidden*2 + feat_dim, 64),
                                  nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 2))
    def seq_repr(self, x, L):
        B, T, _ = x.shape
        packed = nn.utils.rnn.pack_padded_sequence(x, L.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=T)
        fwd = out[torch.arange(B), L-1, :self.hidden]; bwd = out[:, 0, self.hidden:]
        last = torch.cat([fwd, bwd], dim=1)
        mask = torch.arange(T, device=x.device).unsqueeze(0) < L.unsqueeze(1).to(x.device)
        maxp = out.masked_fill(~mask.unsqueeze(-1), -1e9).max(dim=1).values
        return last + maxp
    def forward(self, x, L, f):
        return self.head(torch.cat([self.seq_repr(x, L), f], dim=1))

model = FusionModel(feat_dim=FEAT_DIM).to(device)
pos_w = float((ytr == 0).sum()) / max(1, int((ytr == 1).sum()))
criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_w], dtype=torch.float32).to(device))
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

start_epoch, best_f1, best_ep = 0, 0, 0
if RESUME and os.path.exists(CKPT):
    ck = torch.load(CKPT, map_location='cpu')
    model.load_state_dict(ck['model']); model.to(device)
    start_epoch = ck['epoch']; best_f1 = ck['best_f1']; best_ep = ck['best_ep']
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    for _ in range(start_epoch):
        scheduler.step()
    print(f'Resumed from epoch {start_epoch}, best_f1={best_f1:.4f} (ep{best_ep})', flush=True)

Xtr_t = torch.FloatTensor(Xtr).to(device); ltr_t = torch.LongTensor(ltr); ytr_t = torch.LongTensor(ytr).to(device); Ftr_t = torch.FloatTensor(Ftr).to(device)
Xva_t = torch.FloatTensor(Xva).to(device); lva_t = torch.LongTensor(lva); yva_t = torch.LongTensor(yva).to(device); Fva_t = torch.FloatTensor(Fva).to(device)
Xte_t = torch.FloatTensor(Xte).to(device); lte_t = torch.LongTensor(lte); yte_t = torch.LongTensor(yte).to(device); Fte_t = torch.FloatTensor(Fte).to(device)

from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score)
BATCH = 256
t0 = time.time()
for ep in range(start_epoch, EPOCHS):
    model.train(); perm = torch.randperm(len(Xtr_t)); tot = 0
    for i in range(0, len(perm), BATCH):
        idx = perm[i:i+BATCH]
        out = model(Xtr_t[idx], ltr_t[idx], Ftr_t[idx])
        loss = criterion(out, ytr_t[idx])
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); tot += loss.item()
    scheduler.step(); model.eval()
    with torch.no_grad():
        vo = model(Xva_t, lva_t, Fva_t); vp = vo.argmax(1).cpu().numpy()
        vf1 = f1_score(yva, vp, zero_division=0)
    if vf1 > best_f1:
        best_f1 = vf1; best_ep = ep + 1
        torch.save({k: v.cpu().clone() for k, v in model.state_dict().items()}, f'{OUT}/fusion_model.pt')
        json.dump({'best_epoch': best_ep, 'val_f1': float(best_f1)}, open(f'{OUT}/fusion_best.json', 'w'))
    # checkpoint 当前进度 (含当前模型权重) 供续训
    torch.save({'model': {k: v.cpu().clone() for k, v in model.state_dict().items()},
                'epoch': ep + 1, 'best_f1': best_f1, 'best_ep': best_ep}, CKPT)
    if (ep+1) % 5 == 0 or ep == EPOCHS-1:
        print(f'  Epoch {ep+1}/{EPOCHS} loss={tot:.4f} val_f1={vf1:.4f} best={best_f1:.4f} (ep{best_ep})', flush=True)

print(f'Training loop done ({EPOCHS} epochs) in {time.time()-t0:.0f}s; best val_f1={best_f1:.4f} ep{best_ep}', flush=True)

# 最终评估: 载入最佳模型
if os.path.exists(f'{OUT}/fusion_model.pt'):
    model.load_state_dict(torch.load(f'{OUT}/fusion_model.pt', map_location='cpu')); model.to(device); model.eval()
    with torch.no_grad():
        te_out = model(Xte_t, lte_t, Fte_t)
        te_prob = torch.softmax(te_out, 1)[:, 1].cpu().numpy(); te_pred = te_out.argmax(1).cpu().numpy()
    res = {'Acc': accuracy_score(yte, te_pred), 'Prec': precision_score(yte, te_pred, zero_division=0),
           'Recall': recall_score(yte, te_pred), 'F1': f1_score(yte, te_pred),
           'AUC': roc_auc_score(yte, te_prob), 'PR-AUC': average_precision_score(yte, te_prob)}
    print('\n=== Fusion Model on Test (owners7-8), th=0.5 ===', flush=True)
    for k, v in res.items(): print(f'  {k}: {v:.4f}', flush=True)
    sweep = {}
    print('\n=== Threshold sweep ===', flush=True)
    for th in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        p = (te_prob >= th).astype(int)
        sweep[th] = {'Recall': float(recall_score(yte, p, zero_division=0)), 'Prec': float(precision_score(yte, p, zero_division=0)), 'F1': float(f1_score(yte, p, zero_division=0))}
        print(f'  th={th}: Recall={sweep[th]["Recall"]:.4f} Prec={sweep[th]["Prec"]:.4f} F1={sweep[th]["F1"]:.4f}', flush=True)
    try:
        base = json.load(open(f'{OUT}/routeC_bilstm_results.json'))
        print('\n=== Compare: Fusion vs Pure Bi-LSTM (routeC) ===', flush=True)
        for k in ['Recall','Prec','F1','AUC','PR-AUC']:
            print(f'  {k}: fusion={res[k]:.4f}  pure={base[k]:.4f}  Δ={res[k]-base[k]:+.4f}', flush=True)
    except Exception as e:
        base = None; print('baseline compare skipped:', e)
    out = {'fusion': res, 'sweep': sweep, 'baseline_pure_bilstm': base,
           'meta': {'feat_dim': FEAT_DIM, 'best_epoch': best_ep, 'pos_weight': pos_w, 'device': str(device), 'seed': 42}}
    json.dump(out, open(f'{OUT}/fusion_results.json', 'w'), indent=2)
    np.save(f'{OUT}/fusion_prob.npy', te_prob); np.save(f'{OUT}/fusion_pred.npy', te_pred)
    print('\nSaved fusion_results.json, fusion_prob.npy, fusion_pred.npy', flush=True)
