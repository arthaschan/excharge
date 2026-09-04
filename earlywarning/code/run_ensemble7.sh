#!/bin/bash
# run_ensemble7.sh — Phase 2 方案A: τ=3/5 × 7-seed Token-Attn 训练批处理
# (gate_report_phase2_probe §3 选项A / R6 纪律: 深度成绩=多seed概率集成)
cd "$(dirname "$0")/.."
mkdir -p logs
rm -f logs/train_*.log

run_one() {
  local t=$1 s=$2
  DEVICE=cpu TAU=$t SEED=$s EPOCHS=30 \
    /Users/arthas/.workbuddy/binaries/python/envs/default/bin/python code/train_prefix_tokenattn.py \
    > "logs/train_tau${t}_s${s}.log" 2>&1
  echo "tau${t}_s${s}: $?"
}

export -f run_one
# τ=3 和 τ=5 各 7 个 seed (0-6), 并行 2 (10核 CPU, 每进程~4线程)
for t in 3 5; do for s in 0 1 2 3 4 5 6; do echo "$t $s"; done; done | \
  xargs -P2 -n2 bash -c 'run_one "$0" "$1"'
echo "ALL_DONE"
