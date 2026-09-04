#!/usr/bin/env python3
"""实验A：重建式异常检测 (LSTM-AutoEncoder)。

核心: 只用户正常样本(owner1-6)训练 LSTM-AE 学"正常充电序列"的重建,
用重建误差做异常分数, 在 owner7-8 测试集评估 PR-AUC。无监督, 不依赖异常标签。

异常分数两种口径:
  - score_mean: 逐点重建 MSE 按样本平均(有效长度内)
  - score_max : 逐点重建误差的最大值(对局部异常点更敏感)
输出: docs/recon_results.json, docs/recon_score.npy
"""
import pickle, numpy as np, time, os, warnings, json
warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '4'
import torch, torch.nn as nn
torch.set_num_threads(4)
torch.manual_seed(42); np.random.seed(42)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = _ROOT + '/data/real/'
OUT = _ROOT + '/docs/'
MAXLEN = 200
EPOCHS = int(os.environ.get('EPOCHS', 30))
PATIENCE = int(os.environ.get('PATIENCE', 8))
HIDDEN = 64; N_LAYERS = 2

with open(f'{DATA}/fusion_data.pkl', 'rb') as f:
    D = pickle.load(f)
N_SEQ = len(D['seq_feats'])

def pad(seqs):
    B = len(seqs); X = np.zeros((B, MAXLEN, N_SEQ), dtype=np.float32); L = np.zeros(B, dtype=np.int64)
    for i, s in enumerate(seqs):
        n = min(len(s), MAXLEN); X[i, :n] = s[:n]; L[i] = n
    return X, L

Xtr_all, ltr_all = pad(D['train']['X_tensor']); ytr_all = D['train']['y']
Xva, lva = pad(D['val']['X_tensor']);   yva = D['val']['y']
Xte, lte = pad(D['test']['X_tensor']);  yte = D['test']['y']

normal_mask = ytr_all == 0
Xtr = Xtr_all[normal_mask]; ltr = ltr_all[normal_mask]
print(f'Train normal: {Xtr.shape} | Val: {Xva.shape} | Test: {Xte.shape}', flush=True)
print(f'fault rate: train={ytr_all.mean():.4f} val={yva.mean():.4f} test={yte.mean():.4f}', flush=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device, flush=True)

class LSTM_AE(nn.Module):
    def __init__(self, n_seq=N_SEQ, hidden=HIDDEN, n_layers=N_LAYERS):
        super().__init__()
        self.encoder = nn.LSTM(n_seq, hidden, n_layers, batch_first=True, dropout=0.2)
        self.decoder = nn.LSTM(hidden, hidden, n_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden, n_seq)
    def forward(self, x, L):
        packed = nn.utils.rnn.pack_padded_sequence(x, L.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, c) = self.encoder(packed)
        z = h[-1]                                # [B, hidden]
        dec_in = z.unsqueeze(1).repeat(1, MAXLEN, 1)  # [B, L, hidden]
        dec_out, _ = self.decoder(dec_in)
        return self.fc(dec_out)                  # [B, L, n_seq]

def recon_err(recon, x, L):
    """返回 (mean_mse[B], max_err[B])。"""
    mask = (torch.arange(MAXLEN, device=x.device).unsqueeze(0) < L.unsqueeze(1).to(x.device)).float()
    se = ((recon - x) ** 2).sum(-1) * mask        # [B, L]
    Lf = L.to(x.device).float().clamp(min=1)
    mean_mse = se.sum(1) / Lf / N_SEQ             # 逐点平均 MSE
    err = (recon - x).abs().amax(dim=-1)          # [B, L] 逐点最大通道误差
    max_err = (err * mask).max(dim=1).values      # 样本内最大误差
    return mean_mse, max_err

def main():
    model = LSTM_AE().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    Xtr_t = torch.FloatTensor(Xtr).to(device); ltr_t = torch.LongTensor(ltr)
    Xva_t = torch.FloatTensor(Xva).to(device); lva_t = torch.LongTensor(lva)
    Xte_t = torch.FloatTensor(Xte).to(device); lte_t = torch.LongTensor(lte)
    BATCH = 256
    best_loss, best_state, patience = 1e9, None, 0
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train(); perm = torch.randperm(len(Xtr_t)); tot = 0
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i+BATCH]
            recon = model(Xtr_t[idx], ltr_t[idx])
            loss = recon_err(recon, Xtr_t[idx], ltr_t[idx])[0].mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tot += loss.item()
        sched.step(); model.eval()
        with torch.no_grad():
            va_rec = model(Xva_t, lva_t)
            vmean, _ = recon_err(va_rec, Xva_t, lva_t)
            va_norm = vmean[yva == 0].mean().item() if (yva == 0).sum() > 0 else vmean.mean().item()
        if va_norm < best_loss:
            best_loss = va_norm; patience = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        if (ep + 1) % 5 == 0 or ep == EPOCHS - 1:
            print(f'  ep{ep+1}/{EPOCHS} loss={tot:.3f} val_normal_mse={va_norm:.4f} best={best_loss:.4f}', flush=True)
        if patience >= PATIENCE:
            print(f'  early stop ep{ep+1}', flush=True); break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        te_rec = model(Xte_t, lte_t)
        score_mean, score_max = recon_err(te_rec, Xte_t, lte_t)
        score_mean = score_mean.cpu().numpy(); score_max = score_max.cpu().numpy()
    return score_mean, score_max, best_loss

from sklearn.metrics import average_precision_score, roc_auc_score
sm, sx, best_loss = main()
res = {
    'score_mean': {'PR-AUC': average_precision_score(yte, sm), 'AUC': roc_auc_score(yte, sm)},
    'score_max':  {'PR-AUC': average_precision_score(yte, sx), 'AUC': roc_auc_score(yte, sx)},
    'meta': {'epochs': EPOCHS, 'hidden': HIDDEN, 'n_layers': N_LAYERS, 'best_val_normal_mse': best_loss,
             'n_train_normal': int(Xtr.shape[0]), 'device': str(device)}
}
json.dump(res, open(f'{OUT}/recon_results.json', 'w'), indent=2)
np.save(f'{OUT}/recon_score_mean.npy', sm)
np.save(f'{OUT}/recon_score_max.npy', sx)
print(f'\n=== LSTM-AE 重建式异常检测 ===', flush=True)
print(f'  score_mean  PR-AUC={res["score_mean"]["PR-AUC"]:.4f}  AUC={res["score_mean"]["AUC"]:.4f}', flush=True)
print(f'  score_max   PR-AUC={res["score_max"]["PR-AUC"]:.4f}  AUC={res["score_max"]["AUC"]:.4f}', flush=True)
print('DONE', flush=True)
