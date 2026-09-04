#!/usr/bin/env python3
"""Generate Token-Attn architecture diagram (English labels, consistent with other paper figures).

Output: paper/figures/tokenattn_architecture.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(14, 7))
ax.set_xlim(0, 14); ax.set_ylim(0, 7); ax.axis('off')

def box(x, y, w, h, text, fc='#eef3fb', ec='#2b5b9b', fs=9, bold=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", fc=fc, ec=ec, lw=1.2)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fs,
            fontweight='bold' if bold else 'normal', color='#1a1a1a')

def arrow(p1, p2, color='#444444', lw=1.4):
    a = FancyArrowPatch(p1, p2, arrowstyle='-|>', mutation_scale=14, color=color,
                        lw=lw, shrinkA=2, shrinkB=2)
    ax.add_patch(a)

# left: sequence branch
box(0.3, 5.6, 3.6, 0.8, '6-channel series X in R^{Lx6}\n(V / I / P / GunTemp1 / GunTemp2 / SOC)', fc='#fdeeee', ec='#b3473f', bold=True)
box(0.3, 4.4, 3.6, 0.8, 'BiLSTM encoder\n(hidden=64, 2 layers)')
box(0.3, 3.2, 3.6, 0.8, 'Segment into K=8 bins by\ncharging progress (masked mean pool)')
box(0.3, 2.0, 3.6, 0.8, '8 segment tokens  S in R^{8xd}', fc='#f2eefb', ec='#6a4bb3')

# right: feature branch
box(10.1, 5.6, 3.6, 0.8, '62 handcrafted features f in R^{62}\n(stats / temp / segment / overtemp / battery)', fc='#fdeeee', ec='#b3473f', bold=True)
box(10.1, 4.4, 3.6, 0.8, 'Numeric embed W_e\n+ learnable column embed E_col')
box(10.1, 3.2, 3.6, 0.8, '62 feature tokens  F in R^{62xd}', fc='#f2eefb', ec='#6a4bb3')

# middle: fusion encoder
box(5.3, 4.4, 3.4, 0.9, '[CLS] + 8 seg tokens\n+ 62 feature tokens = 71 tokens', fc='#f2eefb', ec='#6a4bb3', bold=True)
box(5.3, 3.0, 3.4, 0.9, 'Transformer encoder\n(2 layers, d=64, 4 heads)\ncross-modal self-attention', ec='#2b5b9b')
box(5.3, 1.7, 3.4, 0.8, '[CLS] -> MLP\nLayerNorm -> Linear -> ReLU -> Linear(2)')
box(5.3, 0.4, 3.4, 0.8, 'y_hat = softmax(MLP(c))\n(fault / normal)', fc='#eaf7ea', ec='#3a7d44', bold=True)

arrow((3.9, 5.9), (3.9, 5.1))
arrow((3.9, 4.4), (3.9, 4.1))
arrow((3.9, 3.2), (3.9, 2.9))
arrow((3.9, 2.2), (5.3, 4.85))
arrow((10.1, 5.6), (10.1, 5.3))
arrow((10.1, 4.4), (10.1, 4.1))
arrow((10.1, 3.4), (8.7, 4.85))
arrow((7.0, 4.4), (7.0, 4.0))
arrow((7.0, 3.0), (7.0, 2.6))
arrow((7.0, 1.7), (7.0, 1.3))

# interpretability side note
ax.text(13.4, 1.7, 'Multi-view\ninterpretability\n(Sec.3.4/4.3):\n- attention weights\n- permutation\n  importance\n- feature-group\n  ablation', ha='left', va='center', fontsize=8.5, color='#6a4bb3',
        bbox=dict(boxstyle='round,pad=0.4', fc='#f7f4ff', ec='#6a4bb3', lw=1))
arrow((8.7, 2.4), (12.8, 1.8), color='#6a4bb3', lw=1.2)

ax.set_title('Token-Attn: multimodal attention-fusion detector', fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('paper/figures/tokenattn_architecture.png', dpi=200, bbox_inches='tight')
plt.close()
print('Saved paper/figures/tokenattn_architecture.png')
