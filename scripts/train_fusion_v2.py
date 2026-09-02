#!/usr/bin/env python3
"""融合模型 V2 训练 (特征分支独立 MLP)。复用断点续训机制。
输出 docs/fusion_v2_*。LOSS=focal 可选 (默认加权 CE)。
"""
import pickle, numpy as np, time, os, warnings, json
warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '4'
import torch, torch.nn as nn
torch.set_num_threads(4); torch.manual_seed(42); np.random.seed(42)

from fusion_model import FusionModel, FocalLoss

DATA = '/Users/arthas/git/excharge/data/real/'
OUT  = '/Users/arthas/git/excharge/docs/'
CKPT = f'{OUT}/fusion_v2_ckpt.pt'
RESUME = os.environ.get('RESUME', '0') == '1'
EPOCHS = int(os.environ.get('EPOCHS', 30))
LOSS = os.environ.get('LOSS', 'ce')   # 'ce' or 'focal'

with open(f'{DATA}/fusion_data.pkl', 'rb') as f:
    D = pickle.load(f)
FEAT_DIM = D['meta']['n_features']; N_SEQ = len(D['seq_feats']); MAXLEN = 200

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
print('Device:', device, '| LOSS:', LOSS, flush=True)

model = FusionModel(feat_dim=FEAT_DIM).to(device)
nparams = sum(p.numel() for p in model.parameters())
print('Model params:', f'{nparams:,}', flush=True)
pos_w = float((ytr == 0).sum()) / max(1, int((ytr == 1).sum()))
w = torch.tensor([1.0, pos_w], dtype=torch.float32).to(device)
criterion = FocalLoss(gamma=2.0, weight=w) if LOSS == 'focal' else nn.CrossEntropyLoss(weight=w)
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
        torch.save({k: v.cpu().clone() for k, v in model.state_dict().items()}, f'{OUT}/fusion_v2_model.pt')
        json.dump({'best_epoch': best_ep, 'val_f1': float(best_f1)}, open(f'{OUT}/fusion_v2_best.json', 'w'))
    torch.save({'model': {k: v.cpu().clone() for k, v in model.state_dict().items()},
                'epoch': ep + 1, 'best_f1': best_f1, 'best_ep': best_ep}, CKPT)
    if (ep+1) % 5 == 0 or ep == EPOCHS-1:
        print(f'  Epoch {ep+1}/{EPOCHS} loss={tot:.4f} val_f1={vf1:.4f} best={best_f1:.4f} (ep{best_ep})', flush=True)

print(f'Training loop done ({EPOCHS} epochs) in {time.time()-t0:.0f}s; best val_f1={best_f1:.4f} ep{best_ep}', flush=True)

if os.path.exists(f'{OUT}/fusion_v2_model.pt'):
    model.load_state_dict(torch.load(f'{OUT}/fusion_v2_model.pt', map_location='cpu')); model.to(device); model.eval()
    with torch.no_grad():
        te_out = model(Xte_t, lte_t, Fte_t)
        te_prob = torch.softmax(te_out, 1)[:, 1].cpu().numpy(); te_pred = te_out.argmax(1).cpu().numpy()
    res = {'Acc': accuracy_score(yte, te_pred), 'Prec': precision_score(yte, te_pred, zero_division=0),
           'Recall': recall_score(yte, te_pred), 'F1': f1_score(yte, te_pred),
           'AUC': roc_auc_score(yte, te_prob), 'PR-AUC': average_precision_score(yte, te_prob)}
    print('\n=== FusionV2 on Test (owners7-8), th=0.5 ===', flush=True)
    for k, v in res.items(): print(f'  {k}: {v:.4f}', flush=True)
    try:
        base = json.load(open(f'{OUT}/routeC_bilstm_results.json'))
        v1 = json.load(open(f'{OUT}/fusion_results.json'))['fusion']
        print('\n=== Compare: V2 vs V1 vs Pure Bi-LSTM ===', flush=True)
        for k in ['Recall','Prec','F1','AUC','PR-AUC']:
            print(f'  {k}: V2={res[k]:.4f}  V1={v1[k]:.4f}  pure={base[k]:.4f}', flush=True)
    except Exception as e:
        print('compare skipped:', e)
    out = {'fusion_v2': res, 'meta': {'feat_dim': FEAT_DIM, 'best_epoch': best_ep, 'loss': LOSS,
                                      'params': nparams, 'device': str(device)}}
    json.dump(out, open(f'{OUT}/fusion_v2_results.json', 'w'), indent=2)
    np.save(f'{OUT}/fusion_v2_prob.npy', te_prob); np.save(f'{OUT}/fusion_v2_pred.npy', te_pred)
    print('\nSaved fusion_v2_results.json, fusion_v2_prob.npy, fusion_v2_pred.npy', flush=True)
