#!/usr/bin/env python3
"""标准 1D-GradCAM for Bi-LSTM 序列分类（autograd.grad 版，可靠）。
- 前向拆开: 手动跑 LSTM 得到输出序列 A(T,2H), 再 pooling+head
- 时间重要性: alpha_k = GAP(∂y/∂A_k), L_c[t]=ReLU(Σ alpha_k*A[t,k])
- 特征重要性: 输入梯度 ∂y/∂x 逐特征求和
"""
import pickle, numpy as np, time, os, warnings, json
warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '4'
import torch, torch.nn as nn
torch.set_num_threads(4)

DATA = '/Users/arthas/git/excharge/data/real/'
OUT = '/Users/arthas/git/excharge/docs/'

with open(f'{DATA}/seq_tensors.pkl', 'rb') as f:
    d = pickle.load(f)
X_te, y_te = d['X_te'], d['y_te']
FEATS = d['feats']
FEATS_EN = ['Voltage(V)', 'Current(A)', 'Power(kW)', 'GunTemp1(°C)', 'GunTemp2(°C)', 'SOC(%)']

MAXLEN = 200
def pad(seqs):
    B = len(seqs); F = len(FEATS)
    X = np.zeros((B, MAXLEN, F), dtype=np.float32)
    L_arr = np.zeros(B, dtype=np.int64)
    for i, s in enumerate(seqs):
        L = min(len(s), MAXLEN)
        X[i, :L] = s[:L]; L_arr[i] = L
    return X, L_arr

X_te_p, l_te = pad(X_te)
print(f'Test seqs: {len(X_te)} (fault {y_te.sum()})', flush=True)

class BiLSTM(nn.Module):
    def __init__(self, n_feat=6, hidden=64, n_layers=2, dropout=0.2):
        super().__init__()
        self.hidden = hidden
        self.lstm = nn.LSTM(n_feat, hidden, num_layers=n_layers, batch_first=True,
                            bidirectional=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden*2), nn.Linear(hidden*2, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 2))
    def lstm_out(self, x, L):
        B, T, F = x.shape
        packed = nn.utils.rnn.pack_padded_sequence(x, L.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=T)
        return out
    def forward(self, x, L):
        B, T, F = x.shape
        out = self.lstm_out(x, L)
        fwd_last = out[torch.arange(B), L-1, :self.hidden]
        bwd_last = out[:, 0, self.hidden:]
        last = torch.cat([fwd_last, bwd_last], dim=1)
        mask = torch.arange(T, device=x.device).unsqueeze(0) < L.unsqueeze(1).to(x.device)
        out_masked = out.masked_fill(~mask.unsqueeze(-1), -1e9)
        maxp = out_masked.max(dim=1).values
        feat = last + maxp
        return self.head(feat)

model = BiLSTM()
model.load_state_dict(torch.load(f'{OUT}/routeC_bilstm_model.pt', map_location='cpu'))
model.eval()
print('Model loaded.', flush=True)

def gradcam_time(x_t, L_t):
    """标准 GradCAM 时间重要性 (T,)。返回原始 T 长度重要性。"""
    x = x_t.unsqueeze(0).requires_grad_(True)
    l = L_t.unsqueeze(0)
    A = model.lstm_out(x, l)                    # (1,T,2H)
    # 手动 pooling + head 得到 fault logit
    B, T, _ = A.shape
    fwd_last = A[torch.arange(B), l-1, :model.hidden]
    bwd_last = A[:, 0, model.hidden:]
    last = torch.cat([fwd_last, bwd_last], dim=1)
    mask = torch.arange(T, device=x.device).unsqueeze(0) < l.unsqueeze(1).to(x.device)
    A_masked = A.masked_fill(~mask.unsqueeze(-1), -1e9)
    maxp = A_masked.max(dim=1).values
    feat = last + maxp
    logit = model.head(feat)[0, 1]
    # 对 A 和 x 求梯度
    gA, gx = torch.autograd.grad(logit, [A, x], retain_graph=False)
    A = A[0]; gA = gA[0]      # (T,2H)
    alpha = gA.mean(dim=0)    # (2H,)
    Lc = torch.relu((alpha.unsqueeze(0) * A).sum(dim=-1))  # (T,)
    return Lc.detach().numpy(), gx[0].detach().numpy()

def feat_importance(x_t, L_t):
    _, gx = gradcam_time(x_t, L_t)
    return np.abs(gx).sum(axis=0)  # (F,)

X_t = torch.FloatTensor(X_te_p)
L_t = torch.LongTensor(l_te)

fault_idx = np.where(y_te == 1)[0]
normal_idx = np.where(y_te == 0)[0]
print(f'Fault: {len(fault_idx)}, Normal: {len(normal_idx)}', flush=True)

def agg_time(idx_list):
    acc = np.zeros(MAXLEN); n = 0
    for idx in idx_list:
        ti, _ = gradcam_time(X_t[idx], L_t[idx])
        L = l_te[idx]
        acc[:L] += ti[:L]
        n += 1
    return acc / max(n, 1), n

time_fault, nf = agg_time(fault_idx)
time_normal, nn_ = agg_time(normal_idx)
print(f'Aggregated time importance: fault n={nf}, normal n={nn_}', flush=True)

feat_imp = np.zeros(len(FEATS))
for idx in fault_idx:
    feat_imp += feat_importance(X_t[idx], L_t[idx])
feat_imp /= max(len(fault_idx), 1)
order = np.argsort(-feat_imp)
print('\n=== 特征重要性 (输入梯度, 故障样本 n=%d) ===' % len(fault_idx), flush=True)
for j in order:
    print(f'  {FEATS_EN[j]}: {feat_imp[j]:.4f}', flush=True)

np.save(f'{OUT}/routeC_gradcam_time_fault.npy', time_fault)
np.save(f'{OUT}/routeC_gradcam_time_normal.npy', time_normal)
np.save(f'{OUT}/routeC_gradcam_feat_imp.npy', feat_imp)
json.dump({'feats': FEATS, 'feats_en': FEATS_EN, 'feat_imp': feat_imp.tolist(),
           'time_fault': time_fault.tolist(), 'time_normal': time_normal.tolist(),
           'n_fault': nf, 'n_normal': nn_,
           'lens_fault': [int(l_te[i]) for i in fault_idx]},
          open(f'{OUT}/routeC_gradcam_results.json', 'w'), indent=2)
print('Saved standard GradCAM results.', flush=True)
