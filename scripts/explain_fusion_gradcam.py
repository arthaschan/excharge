#!/usr/bin/env python3
"""融合模型的 1D-GradCAM (序列分支): 对 6 通道原始时序做逐通道 x 逐时间步归因。
复用 explain_bilstm_gradcam 的 autograd 方案, 但作用于 FusionModel 的 BiLSTM 序列分支。
输出: docs/fusion_gradcam_time_fault.npy / _normal.npy / _feat_imp.npy / fusion_gradcam_results.json
"""
import pickle, numpy as np, time, os, warnings, json
warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '4'
import torch, torch.nn as nn
torch.set_num_threads(4)

DATA = '/Users/arthas/git/excharge/data/real/'
OUT  = '/Users/arthas/git/excharge/docs/'

with open(f'{DATA}/fusion_data.pkl', 'rb') as f:
    D = pickle.load(f)
SEQ_FEATS = D['seq_feats']
FEATS_EN = ['Voltage(V)', 'Current(A)', 'Power(kW)', 'GunTemp1(°C)', 'GunTemp2(°C)', 'SOC(%)']
MAXLEN = 200

def pad(seqs):
    B = len(seqs); F = len(SEQ_FEATS)
    X = np.zeros((B, MAXLEN, F), dtype=np.float32); L = np.zeros(B, dtype=np.int64)
    for i, s in enumerate(seqs):
        n = min(len(s), MAXLEN); X[i, :n] = s[:n]; L[i] = n
    return X, L

Xte, lte = pad(D['test']['X_tensor']); yte = D['test']['y']
print(f'Test seqs: {len(Xte)} (fault {yte.sum()})', flush=True)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
# 内联 FusionModel 定义 (避免 import train_fusion 触发训练)
import torch.nn as nn
class FusionModel(nn.Module):
    def __init__(self, n_seq=6, hidden=64, n_layers=2, feat_dim=62, dropout=0.2):
        super().__init__()
        self.hidden = hidden
        self.lstm = nn.LSTM(n_seq, hidden, num_layers=n_layers, batch_first=True,
                            bidirectional=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden*2 + feat_dim),
            nn.Linear(hidden*2 + feat_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 2))
    def lstm_out(self, x, L):
        B, T, _ = x.shape
        packed = nn.utils.rnn.pack_padded_sequence(x, L.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=T)
        return out
    def seq_repr(self, x, L):
        B, T, _ = x.shape
        A = self.lstm_out(x, L)
        fwd_last = A[torch.arange(B), L-1, :self.hidden]
        bwd_last = A[:, 0, self.hidden:]
        last = torch.cat([fwd_last, bwd_last], dim=1)
        mask = torch.arange(T, device=x.device).unsqueeze(0) < L.unsqueeze(1).to(x.device)
        Am = A.masked_fill(~mask.unsqueeze(-1), -1e9)
        maxp = Am.max(dim=1).values
        return last + maxp
    def forward(self, x, L, f):
        sr = self.seq_repr(x, L)
        h = torch.cat([sr, f], dim=1)
        return self.head(h)

FEAT_DIM = D['meta']['n_features']
model = FusionModel(feat_dim=FEAT_DIM)
model.load_state_dict(torch.load(f'{OUT}/fusion_model.pt', map_location='cpu'))
model.to(device).eval()
print('Fusion model loaded.', flush=True)

def seq_repr_from_A(A, l, hidden):
    B, T, _ = A.shape
    fwd_last = A[torch.arange(B), l-1, :hidden]
    bwd_last = A[:, 0, hidden:]
    last = torch.cat([fwd_last, bwd_last], dim=1)
    mask = torch.arange(T, device=A.device).unsqueeze(0) < l.unsqueeze(1).to(A.device)
    Am = A.masked_fill(~mask.unsqueeze(-1), -1e9)
    maxp = Am.max(dim=1).values
    return last + maxp

def gradcam_time(x_t, L_t, f_t):
    x = x_t.unsqueeze(0).requires_grad_(True)
    f = f_t.unsqueeze(0); l = L_t.unsqueeze(0)
    A = model.lstm_out(x, l)                       # (1,T,2H)
    sr = seq_repr_from_A(A, l, model.hidden)
    h = torch.cat([sr, f], dim=1)
    logit = model.head(h)[0, 1]
    gA, gx = torch.autograd.grad(logit, [A, x])
    A0, gA0 = A[0], gA[0]
    alpha = gA0.mean(dim=0)
    Lc = torch.relu((alpha.unsqueeze(0) * A0).sum(dim=-1))   # (T,)
    return Lc.detach().cpu().numpy(), gx[0].detach().cpu().numpy()

def feat_importance(x_t, L_t, f_t):
    _, gx = gradcam_time(x_t, L_t, f_t)
    return np.abs(gx).sum(axis=0)

X_t = torch.FloatTensor(Xte).to(device); L_t = torch.LongTensor(lte).to(device)
Fte = torch.FloatTensor(D['test']['X_feat'].astype(np.float32)).to(device)

fault_idx = np.where(yte == 1)[0]; normal_idx = np.where(yte == 0)[0]
print(f'Fault: {len(fault_idx)}, Normal: {len(normal_idx)}', flush=True)

def agg_time(idx_list):
    acc = np.zeros(MAXLEN); n = 0
    for idx in idx_list:
        ti, _ = gradcam_time(X_t[idx], L_t[idx], Fte[idx])
        L = lte[idx]; acc[:L] += ti[:L]; n += 1
    return acc / max(n, 1), n

time_fault, nf = agg_time(fault_idx)
time_normal, nn_ = agg_time(normal_idx)
print(f'Aggregated time importance: fault n={nf}, normal n={nn_}', flush=True)

feat_imp = np.zeros(len(SEQ_FEATS))
for idx in fault_idx:
    feat_imp += feat_importance(X_t[idx], L_t[idx], Fte[idx])
feat_imp /= max(len(fault_idx), 1)
order = np.argsort(-feat_imp)
print('\n=== 序列通道特征重要性 (GradCAM 输入梯度, 故障样本 n=%d) ===' % len(fault_idx), flush=True)
for j in order:
    print(f'  {FEATS_EN[j]}: {feat_imp[j]:.4f}', flush=True)

np.save(f'{OUT}/fusion_gradcam_time_fault.npy', time_fault)
np.save(f'{OUT}/fusion_gradcam_time_normal.npy', time_normal)
np.save(f'{OUT}/fusion_gradcam_feat_imp.npy', feat_imp)
json.dump({'feats': SEQ_FEATS, 'feats_en': FEATS_EN, 'feat_imp': feat_imp.tolist(),
           'time_fault': time_fault.tolist(), 'time_normal': time_normal.tolist(),
           'n_fault': nf, 'n_normal': nn_}, open(f'{OUT}/fusion_gradcam_results.json', 'w'), indent=2)
print('Saved fusion GradCAM results.', flush=True)
