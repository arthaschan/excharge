#!/usr/bin/env python3
"""P1 探路 — Token-Attn 上验证"解释增强特征"（方案A 深度模型侧）

背景：P1 LightGBM 侧已出负结果（62 维已含机制信号，加 5 个比值特征 Δ=-0.006）。
本脚本在深度模型 Token-Attn 上做单 seed 探路（R9 探路纪律，看方向不量产），
判断深度模型是否因"显式比值特征"而受益（树能天然分裂比值，深度特征分支是线性+ReLU，学比值更难）。

口径：与 train_c1c2.py 完全一致（fusion_data.pkl, owner1-6 训/owner7-8 测, seed42, 30 epoch,
      AdamW lr1e-3, weighted CE, batch256）。模型类内嵌（避免 import 训练脚本触发数据加载）。
输出：journal/docs/p1_tokenattn_probe.json
"""
import pickle, numpy as np, time, os, json, warnings
import pandas as pd
warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '4'
import torch, torch.nn as nn
torch.set_num_threads(4)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BASE)
DATA = os.path.join(ROOT, 'data', 'real')
OUT = os.path.join(BASE, 'docs')
os.makedirs(OUT, exist_ok=True)

SEED = int(os.environ.get('SEED', 42))
DEVICE = os.environ.get('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = int(os.environ.get('EPOCHS', 30))
BATCH = int(os.environ.get('BATCH', 64))
MAXLEN = 200
D_MODEL = 64; N_HEADS = 4; N_LAYERS = 2; FF = 128; DROPOUT = 0.1
EPS = 1e-6
XAI_COLS = ['power_decay_ratio', 'power_tail_ratio', 'soc_utilization', 'temp_duration_rate', 'power_onset_ratio']

torch.manual_seed(SEED); np.random.seed(SEED)

# ---------------- 载入 62 维 + 计算 5 个解释特征 ----------------
D = pickle.load(open(f'{DATA}/fusion_data.pkl', 'rb'))
F62 = {k: D[k]['X_feat'].astype(np.float32) for k in ['train', 'val', 'test']}
y = {k: D[k]['y'] for k in ['train', 'val', 'test']}
tx = {k: D[k]['tx'] for k in ['train', 'val', 'test']}
N_SEQ = len(D['seq_feats'])

df = pd.read_parquet(f'{DATA}/all_data.parquet')
grp = {t: s.sort_values('begin_time') for t, s in df.groupby('transaction_id', sort=False)}

def xai_features(sub):
    p = sub['out_power'].to_numpy(dtype=np.float64)
    soc = sub['current_soc'].to_numpy(dtype=np.float64)
    t2 = sub['charging_gun_temperature2'].to_numpy(dtype=np.float64)
    dur = float(sub['total_charging_min'].max()); n = len(p)
    f = {}
    pmax = float(np.nanmax(p)) if n else np.nan
    f['power_decay_ratio'] = (float(p[-1]) / pmax) if (n and np.isfinite(pmax) and pmax > EPS) else np.nan
    tail = p[max(0, 2 * n // 3):]
    f['power_tail_ratio'] = (float(np.nanmean(tail)) / pmax) if (len(tail) and np.isfinite(pmax) and pmax > EPS) else np.nan
    s0 = float(soc[0]) if (n and np.isfinite(soc[0])) else np.nan
    s1 = float(soc[-1]) if (n and np.isfinite(soc[-1])) else np.nan
    denom = 100.0 - s0
    f['soc_utilization'] = ((s1 - s0) / denom) if (np.isfinite(s0) and np.isfinite(s1) and abs(denom) > EPS) else np.nan
    t2max = float(np.nanmax(t2)) if n else np.nan
    f['temp_duration_rate'] = (t2max / dur) if (np.isfinite(t2max) and dur > EPS) else np.nan
    f['power_onset_ratio'] = (float(p[0]) / pmax) if (n and np.isfinite(pmax) and pmax > EPS) else np.nan
    return f

def assemble(tx_list):
    M = np.zeros((len(tx_list), len(XAI_COLS)), dtype=np.float64)
    for i, t in enumerate(tx_list):
        if t in grp:
            fd = xai_features(grp[t])
            for j, c in enumerate(XAI_COLS):
                M[i, j] = fd.get(c, np.nan)
        else:
            M[i, :] = np.nan
    return M

Xai = {k: assemble(tx[k]) for k in ['train', 'val', 'test']}
med = np.nanmedian(Xai['train'], axis=0)
for k in ['train', 'val', 'test']:
    Xai[k] = np.where(np.isnan(Xai[k]), med, Xai[k]).astype(np.float32)
mu = Xai['train'].mean(0); sd = Xai['train'].std(0); sd = np.where(sd < 1e-8, 1.0, sd)
for k in ['train', 'val', 'test']:
    Xai[k] = ((Xai[k] - mu) / sd).astype(np.float32)

F67 = {k: np.hstack([F62[k], Xai[k]]).astype(np.float32) for k in ['train', 'val', 'test']}

# ---------------- 模型类（内嵌自 train_c1c2.py） ----------------
class BiLSTMBackbone(nn.Module):
    def __init__(self, n_seq=6, hidden=64, n_layers=2):
        super().__init__()
        self.hidden = hidden
        self.lstm = nn.LSTM(n_seq, hidden, num_layers=n_layers, batch_first=True, bidirectional=True, dropout=0.2)
    def forward(self, x, L):
        packed = nn.utils.rnn.pack_padded_sequence(x, L.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=MAXLEN)
        return out

def make_head(in_dim, dropout=0.2):
    return nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, 64), nn.ReLU(),
                         nn.Dropout(dropout), nn.Linear(64, 2))

def make_len_mask(L, T, dev):
    return torch.arange(T, device=dev).unsqueeze(0) < L.unsqueeze(1).to(dev)

class TokenAttnFusion(nn.Module):
    def __init__(self, feat_dim=62, hidden=64, d=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, K=8):
        super().__init__()
        self.backbone = BiLSTMBackbone(hidden=hidden)
        self.hidden = hidden; self.K = K
        self.seg_proj = nn.Linear(hidden * 2, d)
        self.feat_embed = nn.Linear(1, d)
        self.col_emb = nn.Parameter(torch.randn(feat_dim, d) * 0.01)
        self.cls = nn.Parameter(torch.randn(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, n_heads, dim_feedforward=FF, dropout=DROPOUT, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.proj = nn.Linear(d, 64)
        self.head = make_head(64)
    def segment_pool(self, A, L):
        B, T, _ = A.shape
        mask = make_len_mask(L, T, A.device).unsqueeze(-1).float()
        Am = A * mask; seg_len = T // self.K; outs = []
        for k in range(self.K):
            seg = Am[:, k * seg_len:(k + 1) * seg_len, :]
            cnt = mask[:, k * seg_len:(k + 1) * seg_len, :].sum(1).clamp(min=1)
            outs.append(seg.sum(1) / cnt)
        return torch.stack(outs, dim=1)
    def forward(self, x, L, f):
        A = self.backbone(x, L)
        seg = self.seg_proj(self.segment_pool(A, L))
        B = f.shape[0]
        ft = self.feat_embed(f.unsqueeze(-1)) + self.col_emb.unsqueeze(0)
        cls = self.cls.expand(B, -1, -1)
        h = torch.cat([cls, seg, ft], dim=1)
        h = self.encoder(h)
        return self.head(self.proj(h[:, 0]))

def pad(seqs):
    B = len(seqs); X = np.zeros((B, MAXLEN, N_SEQ), dtype=np.float32); L = np.zeros(B, dtype=np.int64)
    for i, s in enumerate(seqs):
        n = min(len(s), MAXLEN); X[i, :n] = s[:n]; L[i] = n
    return X, L

from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

Xtr, ltr = pad(D['train']['X_tensor']); ytr = y['train']
Xva, lva = pad(D['val']['X_tensor']);   yva = y['val']
Xte, lte = pad(D['test']['X_tensor']);  yte = y['test']
device = torch.device(DEVICE)
print(f'Device {device} | train {Xtr.shape} | test {Xte.shape} | fault_test {yte.sum()}', flush=True)

def run(feat_dim, Fset, tag):
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = TokenAttnFusion(feat_dim=feat_dim).to(device)
    pos_w = float((ytr == 0).sum()) / max(1, int((ytr == 1).sum()))
    w = torch.tensor([1.0, pos_w], dtype=torch.float32).to(device)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    Ftr_t = torch.FloatTensor(Fset['train']).to(device); Fva_t = torch.FloatTensor(Fset['val']).to(device); Fte_t = torch.FloatTensor(Fset['test']).to(device)
    Xtr_t = torch.FloatTensor(Xtr).to(device); ltr_t = torch.LongTensor(ltr)
    Xva_t = torch.FloatTensor(Xva).to(device); lva_t = torch.LongTensor(lva)
    Xte_t = torch.FloatTensor(Xte).to(device); lte_t = torch.LongTensor(lte)
    ytr_t = torch.LongTensor(ytr).to(device); yva_t = torch.LongTensor(yva).to(device)
    best_f1, best_ep, best_state = 0, 0, None
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train(); perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            if len(idx) < 2:
                continue
            out = model(Xtr_t[idx], ltr_t[idx], Ftr_t[idx])
            loss = crit(out, ytr_t[idx])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step(); model.eval()
        with torch.no_grad():
            vp = model(Xva_t, lva_t, Fva_t).argmax(1).cpu().numpy()
            vf1 = float(f1_score(yva, vp, zero_division=0))
        if vf1 > best_f1:
            best_f1, best_ep = vf1, ep + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        prob = torch.softmax(model(Xte_t, lte_t, Fte_t), 1)[:, 1].cpu().numpy()
    r = {'feat_dim': feat_dim, 'PR-AUC': float(average_precision_score(yte, prob)),
         'AUC': float(roc_auc_score(yte, prob)), 'best_ep': best_ep, 'sec': round(time.time() - t0, 1)}
    print(f'  {tag} (feat_dim={feat_dim}): PR-AUC={r["PR-AUC"]:.4f} AUC={r["AUC"]:.4f} best_ep={best_ep} {r["sec"]}s', flush=True)
    return r, prob

r62, p62 = run(62, F62, '62维')
r67, p67 = run(67, F67, '67维(+5解释特征)')

out = {'meta': {'date': '2026-09-04', 'env': str(device), 'seed': SEED, 'epochs': EPOCHS,
                'protocol': 'owner1-6 train / owner7-8 test, seed42, 单seed探路(R9)',
                'xai_features': XAI_COLS},
       'tokenattn_62': r62, 'tokenattn_67': r67,
       'delta_pr': float(r67['PR-AUC'] - r62['PR-AUC'])}
with open(f'{OUT}/p1_tokenattn_probe.json', 'w', encoding='utf-8') as fp:
    json.dump(out, fp, ensure_ascii=False, indent=2)
np.save(f'{OUT}/p1_tokenattn_62_prob.npy', p62)
np.save(f'{OUT}/p1_tokenattn_67_prob.npy', p67)
print(f'\n=== P1 Token-Attn 探路判据 ===', flush=True)
print(f'  62维 PR-AUC={r62["PR-AUC"]:.4f} | 67维 PR-AUC={r67["PR-AUC"]:.4f} | Δ={out["delta_pr"]:+.4f}', flush=True)
print(f'  (对照: 论文 Token-Attn 7-seed=0.918, 单seed≈0.874±0.033)', flush=True)
print('  结果已保存 journal/docs/p1_tokenattn_probe.json', flush=True)
