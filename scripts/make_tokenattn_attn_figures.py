#!/usr/bin/env python3
"""Token-Attn 注意力可解释性图 (D1 收尾)。

载入 train_c1c2.py 保存的 checkpoint (docs/c1c2_tokenattn[_s{SEED}]_model.pt),
对 owner7-8 测试集批量前向, hook TransformerEncoder 的注意力权重,
生成 4 张注意力可解释性图, 支撑论文叙事:
  "哪个(类)特征 token 在关注充电序列的哪段时间段, 正常 vs 异常如何不同"

输出目录: docs/tokenattn_attn_figs/
用法: [SEED=123] python make_tokenattn_attn_figures.py
"""
import os, sys, pickle, warnings
warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '4'
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_c1c2 as T   # 复用数据载入/pad/TokenAttnFusion(保证结构与训练完全一致)

SEED = int(os.environ.get('SEED', 123))
DATA = T.DATA; OUT = T.OUT
FIGDIR = f'{OUT}/tokenattn_attn_figs'
os.makedirs(FIGDIR, exist_ok=True)
K = T.TokenAttnFusion(62).K  # 8 段
print(f'SEED={SEED} K={K} figdir={FIGDIR}', flush=True)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
torch.manual_seed(SEED); np.random.seed(SEED)

# ---------- 数据 ----------
D = pickle.load(open(f'{DATA}/fusion_data.pkl', 'rb'))
FEAT_COLS = D['feat_cols']
Xte, lte = T.pad(D['test']['X_tensor']); yte = np.asarray(D['test']['y'])
Fte = D['test']['X_feat'].astype(np.float32)
print(f'Test: {Xte.shape} y+={yte.sum()} feat={Fte.shape} n_feat={len(FEAT_COLS)}', flush=True)

# ---------- 模型 + checkpoint ----------
sfix = f'_s{SEED}' if SEED != 42 else ''
ckpt = torch.load(f'{OUT}/c1c2_tokenattn{sfix}_model.pt', map_location='cpu')
print('ckpt meta:', {k: v for k, v in ckpt['meta'].items() if k != 'state'}, flush=True)
model = T.TokenAttnFusion(len(FEAT_COLS)).to(device)
model.load_state_dict(ckpt['state']); model.eval()

# ---------- 抓注意力: monkey-patch self_attn.forward ----------
# PyTorch TransformerEncoderLayer 内部调用 self_attn 时传 need_weights=False,
# 返回 (out, None) -> hook 拿不到权重。包一层 forward 强制 need_weights=True
# + average_attn_weights=False (返回 [B,H,S,S]), 只影响返回值, 不改变计算(推理模式)。
attn_buf = {}
for i, layer in enumerate(model.encoder.layers):
    sa = layer.self_attn
    orig_fwd = sa.forward
    def make_wrapped(idx, orig):
        def wrapped(*args, **kwargs):
            kwargs['need_weights'] = True
            kwargs['average_attn_weights'] = False
            out = orig(*args, **kwargs)
            attn_buf[idx] = out[1].detach().float()   # [B,H,S,S]
            return out
        return wrapped
    sa.forward = make_wrapped(i, orig_fwd)

# ---------- 批量前向聚合 ----------
B = 300
rows = []  # 每样本: cls_w(3), feat2seg_mean(62x8)
with torch.no_grad():
    for s in range(0, len(Xte), B):
        e = min(s + B, len(Xte))
        xb = torch.FloatTensor(Xte[s:e]).to(device)
        lb = torch.LongTensor(lte[s:e])
        fb = torch.FloatTensor(Fte[s:e]).to(device)
        model(xb, lb, fb)
        A = attn_buf[len(model.encoder.layers) - 1]      # 最后层 [B,H,S,S]
        A = A.mean(1).cpu().numpy()                      # [B,S,S] 跨头平均
        S = A.shape[1]
        n_seg = K; n_feat = S - 1 - K
        assert n_feat == len(FEAT_COLS), (n_feat, len(FEAT_COLS))
        a_cls = A[:, 0, :]                               # [B,S] CLS 行
        w_cls = a_cls[:, 0]
        w_seg = a_cls[:, 1:1 + n_seg].sum(1)
        w_feat = a_cls[:, 1 + n_seg:].sum(1)
        f2s = A[:, 1 + n_seg:, 1:1 + n_seg]              # [B,62,8] feat 行 -> seg 列
        for i in range(e - s):
            rows.append((w_cls[i], w_seg[i], w_feat[i], f2s[i]))

cls_w = np.array([r[0] for r in rows]); seg_w = np.array([r[1] for r in rows])
feat_w = np.array([r[2] for r in rows])
F2S = np.stack([r[3] for r in rows], 0)   # [N,62,8]
print(f'agg done: N={len(rows)} F2S={F2S.shape}', flush=True)

# ---------- 特征分组 (与 build_fusion_data 语义一致) ----------
G_TEMPCHG = {'t1_slope', 't2_slope', 't1_max_jump', 't2_max_jump', 't1_std_2nd', 't2_std_2nd'}
G_SEGDIFF = {'v_first_third_min', 'v_sag_from_mean', 'a_last_third_max', 'a_first_last_ratio',
             'p_max_jump', 'soc_rate', 'soc_last_rate', 'v_seg_change_1to2', 'v_seg_change_2to3',
             'a_seg_change_1to2', 'a_seg_change_2to3', 'a_seg3_max',
             'v_last3_vs_first3', 'a_last3_vs_first3', 'p_last3_vs_first3'}
G_OVERTEMP = {'t1_over_40', 't2_over_40', 't1_over_45', 't2_over_45'}
G_BATT = {'bt_LFP', 'bt_NMC', 'bt_LMO', 'bt_LCO', 'bt_LP'}

def group_of(name):
    if name in G_OVERTEMP: return 'Over-temp flags'
    if name in G_TEMPCHG:  return 'Temp change'
    if name in G_SEGDIFF:  return 'Segment/diff'
    if name in G_BATT:     return 'Battery type'
    return 'Basic stats'

GROUPS = ['Basic stats', 'Segment/diff', 'Temp change', 'Over-temp flags', 'Battery type']
gid = np.array([GROUPS.index(group_of(c)) for c in FEAT_COLS])   # [62]
gname = ['Basic stats', 'Segment/diff', 'Temp change', 'Over-temp flags', 'Battery type']
order = np.argsort(gid, kind='stable')   # 行按组排
g_sort = gid[order]
seg_labels = [f'S{k+1}' for k in range(K)]
seg_names = [f'Seg {k+1}' for k in range(K)]
cls_all = np.stack([cls_w, seg_w, feat_w], 1)   # [N,3]

norm_ = yte == 0; abn_ = yte == 1
def grp_avg(F2S_sub):
    out = np.zeros((len(GROUPS), K))
    for g in range(len(GROUPS)):
        m = gid == g
        if m.sum() == 0: continue
        out[g] = F2S_sub[:, m, :].mean(axis=(0, 1))   # 该组特征对 8 段的平均注意力
    return out

f2s_abn = grp_avg(F2S[abn_]); f2s_norm = grp_avg(F2S[norm_])
f2s_all = grp_avg(F2S)

CMAP = 'YlOrRd'
FIGKW = dict(dpi=150, bbox_inches='tight')
seg_pos = [f'{k+1}/8' for k in range(K)]

# ---------- fig1: CLS 注意力区块占比 (正常 vs 异常) ----------
fig, ax = plt.subplots(figsize=(7, 5))
labels = ['CLS self', 'Seq segments', 'Feature tokens']
x = np.arange(3); w = 0.36
v_n = cls_all[norm_].mean(0); v_a = cls_all[abn_].mean(0)
b1 = ax.bar(x - w / 2, v_n, w, label='Normal', color='#4C72B0')
b2 = ax.bar(x + w / 2, v_a, w, label='Fault', color='#C44E52')
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel('Avg attention weight of [CLS] row')
ax.set_title('Token-Attn: what the [CLS] token attends to (last layer, mean heads)')
ax.legend()
for b in list(b1) + list(b2):
    ax.annotate(f'{b.get_height():.3f}', (b.get_x() + b.get_width() / 2, b.get_height()),
                ha='center', va='bottom', fontsize=8)
plt.tight_layout(); plt.savefig(f'{FIGDIR}/fig1_cls_attn_dist.png', **FIGKW); plt.close()

# ---------- fig2: 特征组 × 序列段 热力图 (异常样本) ----------
fig, ax = plt.subplots(figsize=(7.5, 4.6))
im = ax.imshow(f2s_abn, cmap=CMAP, aspect='auto')
ax.set_yticks(range(len(GROUPS))); ax.set_yticklabels(gname)
ax.set_xticks(range(K)); ax.set_xticklabels(seg_pos)
ax.set_xlabel('Charging progress (segment of padded timeline)')
ax.set_title('Avg attention: feature groups → time segments (fault samples)')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
for i in range(len(GROUPS)):
    for j in range(K):
        ax.text(j, i, f'{f2s_abn[i, j]:.3f}', ha='center', va='center', fontsize=7)
plt.tight_layout(); plt.savefig(f'{FIGDIR}/fig2_group_x_seg_fault.png', **FIGKW); plt.close()

# ---------- fig3: 62 特征 × 8 段 热力图 (异常样本, 按组排) ----------
mat = F2S[abn_][:, order, :].mean(0)          # [62,8]
fig, ax = plt.subplots(figsize=(7.5, 15))
im = ax.imshow(mat, cmap=CMAP, aspect='auto')
ax.set_yticks(range(62)); ax.set_yticklabels([FEAT_COLS[i] for i in order], fontsize=6.5)
ax.set_xticks(range(K)); ax.set_xticklabels(seg_pos)
ax.set_xlabel('Charging progress (segment)')
ax.set_title('Per-feature attention → time segments (fault samples)')
plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
# 组色带
gcol = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']
bounds = np.concatenate([[0], np.where(np.diff(g_sort) != 0)[0] + 1, [62]])
for bi in range(len(bounds) - 1):
    ax.axhspan(bounds[bi] - 0.5, bounds[bi + 1] - 0.5, xmin=-0.02, xmax=0, color=gcol[g_sort[bounds[bi]]], lw=0)
ax.set_xlim(-0.6, K - 0.5)
patches = [mpatches.Patch(color=gcol[i], label=gname[i]) for i in range(len(GROUPS))]
ax.legend(handles=patches, loc='lower right', fontsize=8, framealpha=0.9)
plt.tight_layout(); plt.savefig(f'{FIGDIR}/fig3_feat_x_seg_fault.png', **FIGKW); plt.close()

# ---------- fig4: 差异热力图 (异常 - 正常) 特征组×段 ----------
fig, ax = plt.subplots(figsize=(7.5, 4.6))
diff = f2s_abn - f2s_norm
vmax = np.abs(diff).max()
im = ax.imshow(diff, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
ax.set_yticks(range(len(GROUPS))); ax.set_yticklabels(gname)
ax.set_xticks(range(K)); ax.set_xticklabels(seg_pos)
ax.set_xlabel('Charging progress (segment)')
ax.set_title('Attention shift under fault (fault − normal), feature groups → segments')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
for i in range(len(GROUPS)):
    for j in range(K):
        ax.text(j, i, f'{diff[i, j]:+.3f}', ha='center', va='center', fontsize=7)
plt.tight_layout(); plt.savefig(f'{FIGDIR}/fig4_shift_group_x_seg.png', **FIGKW); plt.close()

print('saved:', sorted(os.listdir(FIGDIR)), flush=True)
print('DONE', flush=True)
