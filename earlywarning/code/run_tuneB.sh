#!/bin/bash
# run_tuneB.sh — Phase 2 方案B: τ=3 快速调参诊断 (gate_report_phase2 §5 选项B)
# 目标: 排除"深度落后树 = 欠调参"的可能。单维变体, 与基线(默认配置 seed0 PR-AUC=0.8852)对照。
# 关键: 串行跑(单进程拿满核), 避免 ensemble 2-并行争抢导致的 7h 教训。
cd "$(dirname "$0")/.."
mkdir -p logs
PY=/Users/arthas/.workbuddy/binaries/python/envs/default/bin/python

run_variant() {
  local tag=$1; shift
  # 剩余参数直接作为 env KV 传给训练
  local envs=()
  for kv in "$@"; do envs+=("$kv"); done
  echo "===== $(date +%H:%M:%S) 开始 $tag ====="
  DEVICE=cpu TAU=3 SEED=0 EPOCHS=30 TAG="_tune_$tag" \
    env ${envs[@]} $PY code/train_prefix_tokenattn.py > "logs/tune_$tag.log" 2>&1
  echo "$tag: exit=$? | $(grep 'PR-AUC' logs/tune_$tag.log | head -1)"
}

# 基线(已有): lr1e-3 K8 L2 seed0 = 0.8852 → 无需重跑
# 单维变体: 每次只动一个超参
run_variant lr3e-4  LR=3e-4
run_variant k4      K_SEG=4
run_variant l1      N_LAYERS=1
# 组合: 更小 lr + 更少分段 (常见最优区)
run_variant lr3e-4_k4  LR=3e-4 K_SEG=4
run_variant lr3e-4_l1  LR=3e-4 N_LAYERS=1

echo "ALL_DONE $(date +%H:%M:%S)"
