#!/usr/bin/env python3
"""新模型对比实验：iTransformer / PatchTST / FT-Transformer / DualTransformer。

统一口径: 加权 CE(pos_weight≈20), AdamW lr1e-3, cosine, 30 epoch, seed42, MPS。
数据: data/real/fusion_data.pkl (6通道时序 [L,6] + 62维特征, owner1-6训练/owner7-8测试)。

4 个配置(都保留 62 维手工特征, 只换"编码器"):
  1. iTransformer-Fusion   : 序列分支=6通道token注意力(替换BiLSTM), 特征=MLP
  2. PatchTST-Fusion       : 序列分支=patch注意力(替换BiLSTM), 特征=MLP
  3. FTTransformer-Fusion  : 序列=BiLSTM, 特征分支=62特征token注意力(替换MLP)
  4. DualTransformer-Fusion: 序列=iTransformer + 特征=FTTransformer (双Transformer)

基线: BiLSTM-Fusion PR-AUC=0.748, 纯BiLSTM=0.351, XGBoost特征=0.894
输出: docs/newmodels_{name}_results.json, docs/newmodels_compare.json
"""
import pickle, numpy as np, time, os, warnings, json, sys
warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '4'
import torch, torch.nn as nn
import torch.nn.functional as F
torch.set_num_threads(4)
torch.manual_seed(42); np.random.seed(42)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = _ROOT + '/data/real/'
OUT  = _ROOT + '/docs/'
MAXLEN = 200
EPOCHS = int(os.environ.get('EPOCHS', 30))
PATIENCE = int(os.environ.get('PATIENCE', 10))
D_MODEL = 64; N_HEADS = 4; N_LAYERS = 2; FF = 128; DROPOUT = 0.1

# ---------------- 数据载入 ----------------
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
print(f'Train {Xtr.shape} feat {Ftr.shape} | Val {Xva.shape} | Test {Xte.shape}', flush=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device, '| FEAT_DIM:', FEAT_DIM, '| N_SEQ:', N_SEQ, flush=True)

# ---------------- 编码器 ----------------
class BiLSTMEncoder(nn.Module):
    def __init__(self, n_seq=6, hidden=64, n_layers=2):
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

class iTransformerEncoder(nn.Module):
    """6通道当token, 通道间注意力。输入[B,T,C] -> [B,64]。"""
    def __init__(self, n_seq=6, seq_len=200, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS):
        super().__init__()
        self.embed = nn.Linear(seq_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=FF, dropout=DROPOUT, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.proj = nn.Linear(d_model, 64)
    def forward(self, x, L=None):
        x = x.permute(0, 2, 1)          # [B,C,T]
        h = self.embed(x)               # [B,C,d]
        h = self.encoder(h)             # [B,C,d]
        h = h.mean(dim=1)               # [B,d]
        return self.proj(h)

class PatchTSTEncoder(nn.Module):
    """channel-independent patch。输入[B,T,C] -> [B,64]。"""
    def __init__(self, n_seq=6, seq_len=200, patch_len=16, stride=8, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS):
        super().__init__()
        self.patch_len = patch_len; self.stride = stride
        self.n_patches = (seq_len - patch_len) // stride + 1
        self.embed = nn.Linear(patch_len, d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=FF, dropout=DROPOUT, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.proj = nn.Linear(d_model, 64)
    def forward(self, x, L=None):
        B, T, C = x.shape
        x = x.permute(0, 2, 1).reshape(B * C, T)         # [B*C, T]
        patches = x.unfold(1, self.patch_len, self.stride)  # [B*C, n_patches, patch_len]
        h = self.embed(patches) + self.pos               # [B*C, n_patches, d]
        h = self.encoder(h)                              # [B*C, n_patches, d]
        h = h.mean(dim=1).reshape(B, C, -1).mean(dim=1)  # [B, d]
        return self.proj(h)

class FeatMLP(nn.Module):
    def __init__(self, feat_dim=62, dropout=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout))
    def forward(self, f):
        return self.mlp(f)

class FTTransformerEncoder(nn.Module):
    """62特征当token + [CLS], 特征间注意力。输入[B,62] -> [B,64]。"""
    def __init__(self, feat_dim=62, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS):
        super().__init__()
        self.feat_dim = feat_dim
        self.embed = nn.Linear(1, d_model)                     # 共享数值embed
        self.col_emb = nn.Parameter(torch.randn(feat_dim, d_model) * 0.01)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=FF, dropout=DROPOUT, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.proj = nn.Linear(d_model, 64)
    def forward(self, f):
        B = f.shape[0]
        h = self.embed(f.unsqueeze(-1)) + self.col_emb.unsqueeze(0)  # [B,62,d]
        cls = self.cls.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1)                               # [B,63,d]
        h = self.encoder(h)
        return self.proj(h[:, 0])

# ---------------- 融合头 ----------------
class FusionHead(nn.Module):
    def __init__(self, dropout=0.2):
        super().__init__()
        self.norm = nn.LayerNorm(128)
        self.fc = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 2))
    def forward(self, sr, fr):
        return self.fc(self.norm(torch.cat([sr, fr], dim=1)))

class FusedModel(nn.Module):
    def __init__(self, seq_enc, feat_enc):
        super().__init__()
        self.seq_enc = seq_enc; self.feat_enc = feat_enc; self.head = FusionHead()
    def forward(self, x, L, f):
        return self.head(self.seq_enc(x, L), self.feat_enc(f))

# ---------------- 训练+评估 ----------------
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

def run_one(name, seq_enc, feat_enc):
    torch.manual_seed(42); np.random.seed(42)
    model = FusedModel(seq_enc, feat_enc).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    pos_w = float((ytr == 0).sum()) / max(1, int((ytr == 1).sum()))
    w = torch.tensor([1.0, pos_w], dtype=torch.float32).to(device)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    Xtr_t = torch.FloatTensor(Xtr).to(device); ltr_t = torch.LongTensor(ltr)
    Xva_t = torch.FloatTensor(Xva).to(device); lva_t = torch.LongTensor(lva)
    Xte_t = torch.FloatTensor(Xte).to(device); lte_t = torch.LongTensor(lte)
    Ftr_t = torch.FloatTensor(Ftr).to(device); Fva_t = torch.FloatTensor(Fva).to(device); Fte_t = torch.FloatTensor(Fte).to(device)
    ytr_t = torch.LongTensor(ytr).to(device); yva_t = torch.LongTensor(yva).to(device)

    BATCH = 256
    best_f1, best_ep, best_state, patience = 0, 0, None, 0
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train(); perm = torch.randperm(len(Xtr_t)); tot = 0
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i+BATCH]
            out = model(Xtr_t[idx], ltr_t[idx], Ftr_t[idx])
            loss = crit(out, ytr_t[idx])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tot += loss.item()
        sched.step(); model.eval()
        with torch.no_grad():
            vo = model(Xva_t, lva_t, Fva_t); vp = vo.argmax(1).cpu().numpy()
            vf1 = f1_score(yva, vp, zero_division=0)
        if vf1 > best_f1:
            best_f1 = vf1; best_ep = ep + 1; patience = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        if (ep + 1) % 5 == 0 or ep == EPOCHS - 1:
            print(f'  [{name}] ep{ep+1}/{EPOCHS} loss={tot:.3f} val_f1={vf1:.4f} best={best_f1:.4f}(ep{best_ep})', flush=True)
        if patience >= PATIENCE:
            print(f'  [{name}] early stop at ep{ep+1}', flush=True); break

    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        te = model(Xte_t, lte_t, Fte_t)
        prob = torch.softmax(te, 1)[:, 1].cpu().numpy(); pred = te.argmax(1).cpu().numpy()
    # 阈值校准: 在 val 上最大化 F1 选最优阈值
    with torch.no_grad():
        vprob = torch.softmax(model(Xva_t, lva_t, Fva_t), 1)[:, 1].cpu().numpy()
    best_th, best_cf1 = 0.5, 0
    for th in np.arange(0.05, 0.96, 0.01):
        f1 = f1_score(yva, (vprob >= th).astype(int), zero_division=0)
        if f1 > best_cf1: best_cf1, best_th = f1, float(th)
    cpred = (prob >= best_th).astype(int)
    res = {
        'Acc': accuracy_score(yte, pred), 'Prec': precision_score(yte, pred, zero_division=0),
        'Recall': recall_score(yte, pred), 'F1': f1_score(yte, pred),
        'AUC': roc_auc_score(yte, prob), 'PR-AUC': average_precision_score(yte, prob),
        'best_epoch': best_ep, 'params': nparams, 'best_th': best_th,
        'calib_F1': f1_score(yte, cpred), 'calib_Recall': recall_score(yte, cpred),
        'calib_Prec': precision_score(yte, cpred, zero_division=0),
        'sec': round(time.time() - t0, 1),
    }
    out = {name: res, 'meta': {'feat_dim': FEAT_DIM, 'd_model': D_MODEL, 'n_layers': N_LAYERS,
                               'pos_weight': pos_w, 'device': str(device), 'epochs': EPOCHS}}
    json.dump(out, open(f'{OUT}/newmodels_{name}_results.json', 'w'), indent=2)
    np.save(f'{OUT}/newmodels_{name}_prob.npy', prob)
    print(f'\n=== {name} === (th=0.5)', flush=True)
    for k in ['Acc', 'Prec', 'Recall', 'F1', 'AUC', 'PR-AUC']:
        print(f'  {k}: {res[k]:.4f}', flush=True)
    print(f'  best_th={best_th:.2f} calib_F1={res["calib_F1"]:.4f} (params={nparams:,}, {res["sec"]}s)', flush=True)
    return res

# ---------------- 主流程 ----------------
if __name__ == '__main__':
    configs = [
        ('iTransformer',    iTransformerEncoder(),       FeatMLP(FEAT_DIM)),
        ('PatchTST',        PatchTSTEncoder(),           FeatMLP(FEAT_DIM)),
        ('FTTransformer',   BiLSTMEncoder(),             FTTransformerEncoder(FEAT_DIM)),
        ('DualTransformer', iTransformerEncoder(),       FTTransformerEncoder(FEAT_DIM)),
    ]
    only = os.environ.get('ONLY', '')
    results = {}
    for name, se, fe in configs:
        if only and only not in name:
            continue
        print(f'\n========== Training {name} ==========', flush=True)
        results[name] = run_one(name, se, fe)
    # 汇总对比
    compare = {'baselines': {
        'pure_bilstm': {'PR-AUC': 0.3512, 'F1': 0.3591},
        'bilstm_fusion': {'PR-AUC': 0.7481, 'F1': 0.7043},
        'xgb_features': {'PR-AUC': 0.8940}},
        'models': {k: {m: v[m] for m in ['PR-AUC', 'F1', 'AUC', 'Recall', 'calib_F1', 'best_th']} for k, v in results.items()}}
    json.dump(compare, open(f'{OUT}/newmodels_compare.json', 'w'), indent=2)
    print('\n\n========== 总对比 ==========', flush=True)
    print('  baseline: pure_bilstm PR-AUC=0.3512 | bilstm_fusion PR-AUC=0.7481 | xgb=0.8940', flush=True)
    for k, v in results.items():
        print(f'  {k:16s} PR-AUC={v["PR-AUC"]:.4f}  F1={v["F1"]:.4f}  AUC={v["AUC"]:.4f}  calib_F1={v["calib_F1"]:.4f}', flush=True)
    print('DONE', flush=True)
