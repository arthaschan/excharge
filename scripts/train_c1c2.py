#!/usr/bin/env python3
"""C1/C2 探索：深度融合强化 + 注意力交互融合。

统一口径: 加权 CE(pos_weight≈20), AdamW lr1e-3, cosine, 30 epoch, seed42, MPS。
数据: data/real/fusion_data.pkl (6通道时序 [L,6] + 62维特征, owner1-6训练/owner7-8测试)。
完整训练(无断点续训), 可选早停。

变体(C1 强化 + C2 交互):
  bilstm        : BiLSTM(last+maxp) + FeatMLP + 拼接  —— 基线复现(应≈0.81)
  bilstm_ms     : BiLSTM(max+mean+last 多尺度池化) + FeatMLP —— C1 强化
  bilstm_focal  : BiLSTM(last+maxp) + FeatMLP + FocalLoss   —— C1 强化
  crossattn     : 特征 query 序列 token 交叉注意力            —— C2 交互
  gated         : 门控融合(gate*sr + (1-gate)*fr)            —— C2 交互
  tokenattn     : 分段池化序列token + 特征token 联合self-attn —— C2 交互(多模态Transformer)

用法: ONLY=名字 [EPOCHS=30] [PATIENCE=10] python train_c1c2.py
输出: docs/c1c2_{name}_results.json, docs/c1c2_compare.json
"""
import pickle, numpy as np, time, os, warnings, json, sys
warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '4'
import torch, torch.nn as nn
import torch.nn.functional as F
torch.set_num_threads(4)
SEED = int(os.environ.get('SEED', 42))
torch.manual_seed(SEED); np.random.seed(SEED)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = _ROOT + '/data/real/'
OUT  = _ROOT + '/docs/'
MAXLEN = 200
EPOCHS = int(os.environ.get('EPOCHS', 30))
PATIENCE = int(os.environ.get('PATIENCE', 999))  # 默认无早停(完整训练)
BATCH = int(os.environ.get('BATCH', 256))
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

_dev = os.environ.get('DEVICE', '')
device = torch.device(_dev) if _dev else torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print('Device:', device, '| FEAT_DIM:', FEAT_DIM, '| N_SEQ:', N_SEQ, flush=True)

# ---------------- 基础模块 ----------------
class BiLSTMBackbone(nn.Module):
    """BiLSTM, 输出 token 序列 [B,T,2h] (供后续池化或注意力)。"""
    def __init__(self, n_seq=6, hidden=64, n_layers=2):
        super().__init__()
        self.hidden = hidden
        self.lstm = nn.LSTM(n_seq, hidden, num_layers=n_layers, batch_first=True,
                            bidirectional=True, dropout=0.2)
    def forward(self, x, L):
        packed = nn.utils.rnn.pack_padded_sequence(x, L.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=MAXLEN)
        return out  # [B,T,2h]

class FeatMLP(nn.Module):
    def __init__(self, feat_dim=62, out_dim=64, dropout=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(), nn.Dropout(dropout))
    def forward(self, f):
        return self.mlp(f)

def make_head(in_dim, dropout=0.2):
    return nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, 64), nn.ReLU(),
                         nn.Dropout(dropout), nn.Linear(64, 2))

def make_len_mask(L, T, dev):
    return torch.arange(T, device=dev).unsqueeze(0) < L.unsqueeze(1).to(dev)

# ---------------- C1 变体 ----------------
class BiLSTM_Fusion(nn.Module):
    """基线: last+maxp 池化 + 特征拼接。"""
    def __init__(self, feat_dim=62, hidden=64, pool='maxlast'):
        super().__init__()
        self.backbone = BiLSTMBackbone(hidden=hidden)
        self.hidden = hidden
        self.pool = pool
        if pool == 'multiscale':
            self.seq_proj = nn.Sequential(nn.Linear(hidden * 2 * 3, 64), nn.ReLU())
        else:
            self.seq_proj = nn.Sequential(nn.Linear(hidden * 2, 64), nn.ReLU())
        self.feat_mlp = FeatMLP(feat_dim, 64)
        self.head = make_head(128)
    def seq_repr(self, x, L):
        A = self.backbone(x, L)
        B, T, _ = A.shape
        fwd = A[torch.arange(B), L - 1, :self.hidden]
        bwd = A[:, 0, self.hidden:]
        last = torch.cat([fwd, bwd], dim=1)  # [B,2h]
        mask = make_len_mask(L, T, x.device)
        maxp = A.masked_fill(~mask.unsqueeze(-1), -1e9).max(dim=1).values  # [B,2h]
        if self.pool == 'multiscale':
            meanp = (A * mask.unsqueeze(-1).float()).sum(1) / L.unsqueeze(1).float().clamp(min=1).to(x.device)
            return torch.cat([last, maxp, meanp], dim=1)  # [B,6h]
        return last + maxp  # [B,2h]
    def forward(self, x, L, f):
        sr = self.seq_proj(self.seq_repr(x, L))
        fr = self.feat_mlp(f)
        return self.head(torch.cat([sr, fr], dim=1))

# ---------------- C2 变体 ----------------
class CrossAttnFusion(nn.Module):
    """特征 query 序列 token 的交叉注意力融合。可解释: 注意力权重显示特征关注序列哪段。"""
    def __init__(self, feat_dim=62, hidden=64, d=D_MODEL):
        super().__init__()
        self.backbone = BiLSTMBackbone(hidden=hidden)
        self.hidden = hidden
        self.key_proj = nn.Linear(hidden * 2, d)
        self.val_proj = nn.Linear(hidden * 2, d)
        self.feat_mlp = FeatMLP(feat_dim, d)
        self.seq_pool_proj = nn.Sequential(nn.Linear(hidden * 2, d), nn.ReLU())
        self.head = make_head(d * 3)
        self.d = d
    def forward(self, x, L, f):
        A = self.backbone(x, L)                       # [B,T,2h]
        B, T, _ = A.shape
        K = self.key_proj(A)                          # [B,T,d]
        V = self.val_proj(A)                          # [B,T,d]
        q = self.feat_mlp(f).unsqueeze(1)             # [B,1,d]
        scores = torch.bmm(q, K.transpose(1, 2)) / (self.d ** 0.5)  # [B,1,T]
        mask = make_len_mask(L, T, x.device).unsqueeze(1)           # [B,1,T]
        scores = scores.masked_fill(~mask, -1e9)
        attn = torch.softmax(scores, dim=-1)          # [B,1,T]
        ctx = torch.bmm(attn, V).squeeze(1)           # [B,d]
        # 序列池化表征(供 head 参考)
        fwd = A[torch.arange(B), L - 1, :self.hidden]
        bwd = A[:, 0, self.hidden:]
        seq_pool = self.seq_pool_proj(torch.cat([fwd, bwd], dim=1))
        return self.head(torch.cat([q.squeeze(1), ctx, seq_pool], dim=1))

class GatedFusion(nn.Module):
    """门控融合: gate = sigmoid(W[sr;fr]), out = gate*sr + (1-gate)*fr。"""
    def __init__(self, feat_dim=62, hidden=64):
        super().__init__()
        self.backbone = BiLSTMBackbone(hidden=hidden)
        self.hidden = hidden
        self.seq_proj = nn.Sequential(nn.Linear(hidden * 2, 64), nn.ReLU())
        self.feat_mlp = FeatMLP(feat_dim, 64)
        self.gate = nn.Sequential(nn.Linear(128, 64), nn.Sigmoid())
        self.head = make_head(64)
    def seq_repr(self, x, L):
        A = self.backbone(x, L)
        B, T, _ = A.shape
        fwd = A[torch.arange(B), L - 1, :self.hidden]
        bwd = A[:, 0, self.hidden:]
        last = torch.cat([fwd, bwd], dim=1)
        mask = make_len_mask(L, T, x.device)
        maxp = A.masked_fill(~mask.unsqueeze(-1), -1e9).max(dim=1).values
        return last + maxp
    def forward(self, x, L, f):
        sr = self.seq_proj(self.seq_repr(x, L))       # [B,64]
        fr = self.feat_mlp(f)                         # [B,64]
        g = self.gate(torch.cat([sr, fr], dim=1))     # [B,64]
        fused = g * sr + (1 - g) * fr                 # [B,64]
        return self.head(fused)

class TokenAttnFusion(nn.Module):
    """多模态Transformer: 分段池化序列token(K段) + 特征token(62) + [CLS] 联合self-attn。"""
    def __init__(self, feat_dim=62, hidden=64, d=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, K=8):
        super().__init__()
        self.backbone = BiLSTMBackbone(hidden=hidden)
        self.hidden = hidden
        self.K = K
        self.seg_proj = nn.Linear(hidden * 2, d)
        self.feat_embed = nn.Linear(1, d)             # 数值 embed
        self.col_emb = nn.Parameter(torch.randn(feat_dim, d) * 0.01)
        self.cls = nn.Parameter(torch.randn(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, n_heads, dim_feedforward=FF, dropout=DROPOUT, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.proj = nn.Linear(d, 64)
        self.head = make_head(64)
    def segment_pool(self, A, L):
        B, T, _ = A.shape
        mask = make_len_mask(L, T, A.device).unsqueeze(-1).float()  # [B,T,1]
        Am = A * mask
        seg_len = T // self.K
        outs = []
        for k in range(self.K):
            seg = Am[:, k*seg_len:(k+1)*seg_len, :]                 # [B,seg_len,d]
            cnt = mask[:, k*seg_len:(k+1)*seg_len, :].sum(1).clamp(min=1)  # [B,1]
            outs.append(seg.sum(1) / cnt)                            # [B,2h]
        return torch.stack(outs, dim=1)                              # [B,K,2h]
    def forward(self, x, L, f):
        A = self.backbone(x, L)
        seg = self.seg_proj(self.segment_pool(A, L))                 # [B,K,d]
        B = f.shape[0]
        ft = self.feat_embed(f.unsqueeze(-1)) + self.col_emb.unsqueeze(0)  # [B,62,d]
        cls = self.cls.expand(B, -1, -1)
        h = torch.cat([cls, seg, ft], dim=1)                         # [B,1+K+62,d]
        h = self.encoder(h)
        return self.head(self.proj(h[:, 0]))

# ---------------- 训练+评估 ----------------
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

def run_one(name, model_factory, loss_type='ce'):
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = model_factory().to(device)
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

    def focal_loss(logits, target, gamma=2.0):
        ce = F.cross_entropy(logits, target, reduction='none', weight=w)
        pt = torch.exp(-ce)
        return ((1 - pt) ** gamma * ce).mean()

    best_f1, best_ep, best_state, patience = 0, 0, None, 0
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train(); perm = torch.randperm(len(Xtr_t)); tot = 0
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i+BATCH]
            if len(idx) < 2:
                continue  # BatchNorm1d 需要 batch>=2，跳过最后单样本 batch
            out = model(Xtr_t[idx], ltr_t[idx], Ftr_t[idx])
            loss = focal_loss(out, ytr_t[idx]) if loss_type == 'focal' else crit(out, ytr_t[idx])
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
    out = {name: res, 'meta': {'feat_dim': FEAT_DIM, 'loss': loss_type,
                               'pos_weight': pos_w, 'device': str(device), 'epochs': EPOCHS}}
    sfix = f'_s{SEED}' if SEED != 42 else ''
    json.dump(out, open(f'{OUT}/c1c2_{name}{sfix}_results.json', 'w'), indent=2)
    np.save(f'{OUT}/c1c2_{name}{sfix}_prob.npy', prob)
    # 保存 best checkpoint (供可解释性图/推理复用)
    torch.save({'state': best_state, 'meta': {'name': name, 'seed': SEED,
                'feat_dim': FEAT_DIM, 'best_epoch': best_ep, 'best_f1': best_f1,
                'PR-AUC': res['PR-AUC'], 'device': str(device)}},
               f'{OUT}/c1c2_{name}{sfix}_model.pt')
    print(f'  [saved] c1c2_{name}{sfix}_model.pt', flush=True)
    print(f'\n=== {name} === (th=0.5)', flush=True)
    for k in ['Acc', 'Prec', 'Recall', 'F1', 'AUC', 'PR-AUC']:
        print(f'  {k}: {res[k]:.4f}', flush=True)
    print(f'  best_th={best_th:.2f} calib_F1={res["calib_F1"]:.4f} (params={nparams:,}, {res["sec"]}s)', flush=True)
    return res

# ---------------- 主流程 ----------------
FACTORIES = {
    'bilstm':       lambda: BiLSTM_Fusion(FEAT_DIM),
    'bilstm_ms':    lambda: BiLSTM_Fusion(FEAT_DIM, pool='multiscale'),
    'bilstm_focal': lambda: BiLSTM_Fusion(FEAT_DIM),
    'crossattn':    lambda: CrossAttnFusion(FEAT_DIM),
    'gated':        lambda: GatedFusion(FEAT_DIM),
    'tokenattn':    lambda: TokenAttnFusion(FEAT_DIM),
}
LOSS_MAP = {'bilstm_focal': 'focal'}

if __name__ == '__main__':
    only = os.environ.get('ONLY', '')
    results = {}
    for name, factory in FACTORIES.items():
        if only and only != name:
            continue
        print(f'\n========== Training {name} ==========', flush=True)
        results[name] = run_one(name, factory, LOSS_MAP.get(name, 'ce'))
    if len(results) > 1:
        compare = {'models': {k: {m: v[m] for m in ['PR-AUC', 'F1', 'AUC', 'Recall', 'calib_F1', 'best_th', 'best_epoch', 'params']} for k, v in results.items()}}
        json.dump(compare, open(f'{OUT}/c1c2_compare.json', 'w'), indent=2)
        print('\n\n========== 总对比 ==========', flush=True)
        for k, v in results.items():
            print(f'  {k:16s} PR-AUC={v["PR-AUC"]:.4f}  F1={v["F1"]:.4f}  AUC={v["AUC"]:.4f}  calib_F1={v["calib_F1"]:.4f}', flush=True)
    print('DONE', flush=True)
