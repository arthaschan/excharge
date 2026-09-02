#!/usr/bin/env python3
"""实验B：域自适应 (CORAL 特征对齐)。

在融合模型(BiLSTM + FeatMLP)上加 CORAL loss, 对齐源域(owner1-6, 有标签)与
目标域(owner7-8, 无标签)的融合表征 z 的二阶统计量(协方差)。
这是 transductive 无监督域自适应: 目标域只用无标签数据做特征对齐, 不碰标签。

对比多个 lambda: lam=0(纯分类=基线), 0.1, 0.5, 1.0。
评估: owner7-8 测试集 PR-AUC(与基线同口径)。
输出: docs/coral_results.json
"""
import pickle, numpy as np, time, os, warnings, json
warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '4'
import torch, torch.nn as nn
torch.set_num_threads(4)
torch.manual_seed(42); np.random.seed(42)

DATA = '/Users/arthas/git/excharge/data/real/'
OUT = '/Users/arthas/git/excharge/docs/'
MAXLEN = 200
EPOCHS = int(os.environ.get('EPOCHS', 30))
PATIENCE = int(os.environ.get('PATIENCE', 10))
D_MODEL = 64; N_HEADS = 4; N_LAYERS = 2; FF = 128; DROPOUT = 0.1

with open(f'{DATA}/fusion_data.pkl', 'rb') as f:
    D = pickle.load(f)
N_SEQ = len(D['seq_feats']); FEAT_DIM = D['meta']['n_features']

def pad(seqs):
    B = len(seqs); X = np.zeros((B, MAXLEN, N_SEQ), dtype=np.float32); L = np.zeros(B, dtype=np.int64)
    for i, s in enumerate(seqs):
        n = min(len(s), MAXLEN); X[i, :n] = s[:n]; L[i] = n
    return X, L

Xtr, ltr = pad(D['train']['X_tensor']); ytr = D['train']['y']
Xva, lva = pad(D['val']['X_tensor']);   yva = D['val']['y']
Xte, lte = pad(D['test']['X_tensor']);  yte = D['test']['y']
Ftr = D['train']['X_feat'].astype(np.float32)
Fva = D['val']['X_feat'].astype(np.float32)
Fte = D['test']['X_feat'].astype(np.float32)
print(f'Train {Xtr.shape} | Val {Xva.shape} | Test {Xte.shape} | FEAT_DIM {FEAT_DIM}', flush=True)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print('Device:', device, flush=True)

class BiLSTMEncoder(nn.Module):
    def __init__(self, n_seq=N_SEQ, hidden=64, n_layers=2):
        super().__init__()
        self.hidden = hidden
        self.lstm = nn.LSTM(n_seq, hidden, num_layers=n_layers, batch_first=True, bidirectional=True, dropout=0.2)
        self.proj = nn.Linear(hidden * 2, 64)
    def lstm_out(self, x, L):
        packed = nn.utils.rnn.pack_padded_sequence(x, L.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=MAXLEN)
        return out
    def forward(self, x, L):
        A = self.lstm_out(x, L)
        fwd = A[torch.arange(len(x)), L - 1, :self.hidden]
        bwd = A[:, 0, self.hidden:]
        last = torch.cat([fwd, bwd], dim=1)
        mask = torch.arange(MAXLEN, device=x.device).unsqueeze(0) < L.unsqueeze(1).to(x.device)
        maxp = A.masked_fill(~mask.unsqueeze(-1), -1e9).max(dim=1).values
        return self.proj(last + maxp)

class FeatMLP(nn.Module):
    def __init__(self, feat_dim=FEAT_DIM, dropout=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout))
    def forward(self, f):
        return self.mlp(f)

class CoralModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq_enc = BiLSTMEncoder()
        self.feat_enc = FeatMLP(FEAT_DIM)
        self.norm = nn.LayerNorm(128)
        self.fc = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 2))
    def get_z(self, x, L, f):
        return self.norm(torch.cat([self.seq_enc(x, L), self.feat_enc(f)], dim=1))
    def forward(self, x, L, f):
        return self.fc(self.get_z(x, L, f))

def coral_loss(a, b):
    d = a.size(1); n1 = a.size(0); n2 = b.size(0)
    a = a - a.mean(0, keepdim=True); b = b - b.mean(0, keepdim=True)
    ca = a.t() @ a / (n1 - 1); cb = b.t() @ b / (n2 - 1)
    return ((ca - cb) ** 2).sum() / (4 * d * d)

from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

Xtr_t = torch.FloatTensor(Xtr).to(device); ltr_t = torch.LongTensor(ltr)
Xva_t = torch.FloatTensor(Xva).to(device); lva_t = torch.LongTensor(lva)
Xte_t = torch.FloatTensor(Xte).to(device); lte_t = torch.LongTensor(lte)
Ftr_t = torch.FloatTensor(Ftr).to(device); Fva_t = torch.FloatTensor(Fva).to(device); Fte_t = torch.FloatTensor(Fte).to(device)
ytr_t = torch.LongTensor(ytr).to(device); yva_t = torch.LongTensor(yva).to(device)

def run_lam(lam):
    torch.manual_seed(42); np.random.seed(42)
    model = CoralModel().to(device)
    pos_w = float((ytr == 0).sum()) / max(1, int((ytr == 1).sum()))
    crit = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_w], dtype=torch.float32).to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    BATCH = 256
    best_f1, best_state, patience = 0, None, 0
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train(); perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(perm), BATCH):
            sidx = perm[i:i+BATCH]
            tidx = torch.randint(0, len(Xte_t), (len(sidx),))
            z_s = model.get_z(Xtr_t[sidx], ltr_t[sidx], Ftr_t[sidx])
            # 目标域前向时冻结 feat_enc 的 BN running stats, 防止 test 数据分布泄露
            model.feat_enc.eval()
            z_t = model.get_z(Xte_t[tidx], lte_t[tidx], Fte_t[tidx])
            model.feat_enc.train()
            out_s = model.fc(z_s)
            loss = crit(out_s, ytr_t[sidx])
            if lam > 0:
                loss = loss + lam * coral_loss(z_s, z_t)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step(); model.eval()
        with torch.no_grad():
            vo = model(Xva_t, lva_t, Fva_t); vf1 = f1_score(yva, vo.argmax(1).cpu().numpy(), zero_division=0)
        if vf1 > best_f1:
            best_f1 = vf1; patience = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        if (ep + 1) % 10 == 0 or ep == EPOCHS - 1:
            print(f'  [lam={lam}] ep{ep+1}/{EPOCHS} val_f1={vf1:.4f} best={best_f1:.4f}', flush=True)
        if patience >= PATIENCE:
            break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        prob = torch.softmax(model(Xte_t, lte_t, Fte_t), 1)[:, 1].cpu().numpy()
    return {'PR-AUC': average_precision_score(yte, prob), 'AUC': roc_auc_score(yte, prob),
            'F1_05': f1_score(yte, (prob >= 0.5).astype(int), zero_division=0),
            'best_val_f1': best_f1, 'sec': round(time.time() - t0, 1)}

lams = [float(x) for x in os.environ.get('LAMS', '0,0.1,0.5,1.0').split(',')]
results = {}
for lam in lams:
    print(f'\n===== CORAL lam={lam} =====', flush=True)
    results[str(lam)] = run_lam(lam)

json.dump(results, open(f'{OUT}/coral_results.json', 'w'), indent=2)
print('\n=== CORAL 域自适应 (owner7-8 测试) ===', flush=True)
for k, v in results.items():
    print(f'  lam={k:5s} PR-AUC={v["PR-AUC"]:.4f}  AUC={v["AUC"]:.4f}  F1_05={v["F1_05"]:.4f}  (val_f1={v["best_val_f1"]:.4f})', flush=True)
print('DONE', flush=True)
