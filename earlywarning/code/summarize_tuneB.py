#!/usr/bin/env python3
"""summarize_tuneB.py — 汇总方案B调参结果, 对照默认配置基线
用法: python summarize_tuneB.py
基线: 默认配置(lr1e-3/K8/L2) seed0 = 0.8852 (docs/prefix_tokenattn_tau3_s0_results.json)
"""
import json, os

OUT = 'docs'
BASELINE = 0.8852  # 默认配置 seed0 (ensemble7 训练之一)
VARIANTS = [
    # (tag, 说明)
    ('lr3e-4',      'LR 1e-3→3e-4'),
    ('k4',          'K_SEG 8→4'),
    ('l1',          'N_LAYERS 2→1'),
    ('lr3e-4_k4',   'LR+K 组合'),
    ('lr3e-4_l1',   'LR+L 组合'),
]

print(f'{"变体":<12} {"说明":<16} {"PR-AUC":>8} {"Δvs基线":>8}  判定')
print('-' * 66)
print(f'{"(基线)":<12} {"lr1e-3 K8 L2":<16} {BASELINE:>8.4f} {"—":>8}  —')
for tag, desc in VARIANTS:
    fp = f'{OUT}/prefix_tokenattn_tau3_tune_{tag}_results.json'
    if not os.path.exists(fp):
        print(f'{tag:<12} {desc:<16} {"缺失":>8}')
        continue
    r = json.load(open(fp))['result']
    pr = r['PR-AUC']
    d = pr - BASELINE
    verdict = ''
    if pr > 0.92:
        verdict = '★ 显著超基线嫌疑 → 扩7-seed复核'
    elif pr > 0.90:
        verdict = '? 略升, 在单seed波动内(±0.06)'
    else:
        verdict = '基线内/低于 → 支持结构性结论'
    print(f'{tag:<12} {desc:<16} {pr:>8.4f} {d:+8.4f}  {verdict}')

print()
print('判读规则: 单seed波动 ±0.06 (τ=3 集成范围 0.792~0.908)')
print('若全部变体 ≤0.90 → 欠调参排除, 结构性结论坐实 → 转 C')
print('若某变体 >0.92 → 欠调参嫌疑, 扩该变体 7-seed 再裁决')
