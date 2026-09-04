#!/usr/bin/env python3
"""P2/C1 — 归因一致性正则（方案C 变体1）

把"训练域归因先验"(LightGBM gain 重要性) 作为软目标，约束 Token-Attn 的
[CLS]→特征 token 注意力分布与之对齐（可微正则），看是否提升检测 PR-AUC。

P2 前提检查已通过：训练域 vs 测试域归因先验 Spearman ρ=0.64、Top-10 Jaccard=0.40，
先验基本跨站迁移（见 p2_attribution_prior_check.json）。

实现：
  - 先验 p_prior = normalize(LightGBM gain importance) ∈ R^62 (sum=1)，在训练域上训一个 LightGBM 得到。
  - 模型可微归因 a_model = Token-Attn 最后一层 [CLS] 对 62 个特征 token 的注意力
    （手写 encoder 前向逐层拿 need_weights=True，规避 torch≥2.x 默认不返权重的坑），
    对 62 维重新归一化 → a_model_norm (sum=1)。
  - 正则 loss_attn = -(p_prior * log(a_model_norm + eps)).sum()  (cross-entropy = KL 减常数)
  - 总 loss = CE + λ * loss_attn，λ ∈ {0, 0.1, 0.5, 1.0} 做消融。
  - 单 seed 42 探路，BATCH=64（复现论文 0.918 的关键），30 epoch。

判据：λ>0 的 PR-AUC 显著高于 λ=0（基线），且归因一致性（KL 下降 / Spearman 上升）同步改善。
输出：journal/docs/p2_c1_attribution_regularization.json
"""
import pickle, numpy as np, time, os, json, warnings
import pandas as pd
warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '4'
import torch, torch.nn as nn
import torch.nn.functional as F
torch.set_num_threads(4)

import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(BASE)
DATA = os.path.join(ROOT, 'data', 'real')
OUT = os.path.join(BASE, 'docs')
os.makedirs(OUT, exist_ok=True)

SEED = int(os.environ.get('SEED', 42))
DEVICE = os.environ.get('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = int(os.environ.get('EPOCHS', 30))
BATCH = int(os.environ.get('BATCH', 64))
LAMBDAS = [float(x) for x in os.environ.get('LAMBDAS', '0,0.1,0.5,1.0').split(',')]
MAXLEN = 200
D_MODEL = 64; N_HEADS = 4; N_LAYERS = 2; FF = 128; DROPOUT = 0.1
EPS = 1e-8

torch.manual_seed(SEED); np.random.seed(SEED)

# ---------------- 载入数据 ----------------
D = pickle.load(open(f'{DATA}/fusion_data.pkl', 'rb'))
cols = D['feat_cols']
N_SEQ = len(D['seq_feats'])
Ftr = D['train']['X_feat'].astype(np.float32); ytr = D['train']['y']
Fva = D['val']['X_feat'].astype(np.float32);   yva = D['val']['y']
Fte = D['test']['X_feat'].astype(np.float32);  yte = D['test']['y']

def pad(seqs):
    Bn = len(seqs); X = np.zeros((Bn, MAXLEN, N_SEQ), dtype=np.float32); L = np.zeros(Bn, dtype=np.int64)
    for i, s in enumerate(seqs):
        n = min(len(s), MAXLEN); X[i, :n] = s[:n]; L[i] = n
    return X, L
Xtr, ltr = pad(D['train']['X_tensor']); Xva, lva = pad(D['val']['X_tensor']); Xte, lte = pad(D['test']['X_tensor'])

# ---------------- 训练域归因先验 ----------------
m_lgb = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=31,
                           subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=4, verbosity=-1)
m_lgb.fit(Ftr, ytr, eval_set=[(Fva, yva)], eval_metric='auc', callbacks=[lgb.early_stopping(100, verbose=False)])
gain = m_lgb.booster_.feature_importance(importance_type='gain').astype(np.float64)
p_prior = gain / (gain.sum() + EPS)                       # [62] sum=1 软先验
p_prior_t = torch.FloatTensor(p_prior).to(torch.device(DEVICE))
print(f'先验 top5: {[cols[i] for i in np.argsort(-gain)[:5]]}', flush=True)

# ---------------- 模型（内嵌，带注意力抓取） ----------------
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

class TokenAttnC1(nn.Module):
    def __init__(self, feat_dim=62, hidden=64, d=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS, K=8):
        super().__init__()
        self.backbone = BiLSTMBackbone(hidden=hidden)
        self.hidden = hidden; self.K = K; self.d = d
        self.seg_proj = nn.Linear(hidden * 2, d)
        self.feat_embed = nn.Linear(1, d)
        self.col_emb = nn.Parameter(torch.randn(feat_dim, d) * 0.01)
        self.cls = nn.Parameter(torch.randn(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, n_heads, dim_feedforward=FF, dropout=DROPOUT, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.proj = nn.Linear(d, 64)
        self.head = make_head(64)
        self.feat_dim = feat_dim
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
        h = torch.cat([cls, seg, ft], dim=1)                 # [B, 1+8+62, d]
        # 手写 encoder 前向，逐层拿最后一层注意力权重（need_weights=True）
        x = h; attn_w = None
        for layer in self.encoder.layers:
            aout, aw = layer.self_attn(x, x, x, need_weights=True, average_attn_weights=True)
            x = layer.norm1(x + layer.dropout1(aout))
            x = layer.norm2(x + layer.dropout2(layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))))
            attn_w = aw                                       # [B, 71, 71]
        feat_attn = attn_w[:, 0, 1 + self.K:]                 # [CLS]→62 特征 token, [B, 62]
        a_model_norm = feat_attn / (feat_attn.sum(1, keepdim=True) + EPS)
        logits = self.head(self.proj(x[:, 0]))
        return logits, a_model_norm

device = torch.device(DEVICE)
Xtr_t = torch.FloatTensor(Xtr).to(device); ltr_t = torch.LongTensor(ltr)
Xva_t = torch.FloatTensor(Xva).to(device); lva_t = torch.LongTensor(lva)
Xte_t = torch.FloatTensor(Xte).to(device); lte_t = torch.LongTensor(lte)
Ftr_t = torch.FloatTensor(Ftr).to(device); Fva_t = torch.FloatTensor(Fva).to(device); Fte_t = torch.FloatTensor(Fte).to(device)
ytr_t = torch.LongTensor(ytr).to(device); yva_t = torch.LongTensor(yva).to(device)
print(f'Device {device} | λ={LAMBDAS} | train {Xtr_t.shape}', flush=True)

def run(lam):
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = TokenAttnC1(feat_dim=len(cols)).to(device)
    pos_w = float((ytr == 0).sum()) / max(1, int((ytr == 1).sum()))
    w = torch.tensor([1.0, pos_w], dtype=torch.float32).to(device)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    best_f1, best_ep, best_state = 0, 0, None
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train(); perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            if len(idx) < 2:
                continue
            logits, a = model(Xtr_t[idx], ltr_t[idx], Ftr_t[idx])
            ce = crit(logits, ytr_t[idx])
            if lam > 0:
                attn_loss = -(p_prior_t.unsqueeze(0) * torch.log(a + EPS)).sum(1).mean()
                loss = ce + lam * attn_loss
            else:
                loss = ce
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step(); model.eval()
        with torch.no_grad():
            vo, _ = model(Xva_t, lva_t, Fva_t)
            vf1 = float(f1_score(yva, vo.argmax(1).cpu().numpy(), zero_division=0))
        if vf1 > best_f1:
            best_f1, best_ep = vf1, ep + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        te, a_te = model(Xte_t, lte_t, Fte_t)
        prob = torch.softmax(te, 1)[:, 1].cpu().numpy()
        a_mean = a_te.mean(0).cpu().numpy()                    # 测试集平均特征注意力
    a_norm = a_mean / (a_mean.sum() + EPS)
    kl = float((p_prior * np.log((p_prior + EPS) / (a_norm + EPS))).sum())
    rho, _ = spearmanr(p_prior, a_norm)
    r = {'lambda': lam, 'PR-AUC': float(average_precision_score(yte, prob)),
         'AUC': float(roc_auc_score(yte, prob)), 'best_ep': best_ep,
         'kl_to_prior': kl, 'spearman_to_prior': float(rho), 'sec': round(time.time() - t0, 1)}
    print(f'  λ={lam}: PR-AUC={r["PR-AUC"]:.4f} AUC={r["AUC"]:.4f} KL={kl:.4f} ρ={rho:.3f} best_ep={best_ep} {r["sec"]}s', flush=True)
    return r, prob, a_norm

results = {}
probs = {}
for lam in LAMBDAS:
    results[str(lam)], probs[str(lam)], _ = run(lam)

base = results['0.0']['PR-AUC']
summary = {'meta': {'date': '2026-09-04', 'env': str(device), 'seed': SEED, 'batch': BATCH,
                    'lambdas': LAMBDAS, 'epochs': EPOCHS,
                    'prior': [cols[i] for i in np.argsort(-gain)[:10]],
                    'protocol': 'owner1-6 train / owner7-8 test, 单seed探路, BATCH=64'},
           'per_lambda': results,
           'delta_vs_baseline': {k: float(v['PR-AUC'] - base) for k, v in results.items()}}
with open(f'{OUT}/p2_c1_attribution_regularization.json', 'w', encoding='utf-8') as fp:
    json.dump(summary, fp, ensure_ascii=False, indent=2)
for k, p in probs.items():
    np.save(f'{OUT}/p2_c1_prob_lam{k}.npy', p)

print(f'\n=== C1 判据（baseline λ=0 PR-AUC={base:.4f}）===', flush=True)
for k, v in results.items():
    d = v['PR-AUC'] - base
    print(f'  λ={k:>4}: PR-AUC={v["PR-AUC"]:.4f} (Δ{d:+.4f}) KL={v["kl_to_prior"]:.4f} ρ={v["spearman_to_prior"]:.3f}', flush=True)
print('  结果已保存 journal/docs/p2_c1_attribution_regularization.json', flush=True)
