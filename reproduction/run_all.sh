#!/usr/bin/env bash
# =============================================================================
# excharge 论文复现一键脚本（R2 主线）
# 在 reproduction/ 根目录执行:  bash run_all.sh
# 依赖: 已装好 torch + requirements.txt（见 README「环境初始化」）
# 数据: data/real/ 已内置 fusion_data.pkl / all_data.parquet / seq_tensors.pkl，
#       如需从原始 xlsx 重建，见 README 4.0（默认跳过）。
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
mkdir -p docs figures

PY="${PYTHON:-python}"
export DEVICE="${DEVICE:-cuda}"
export OMP_NUM_THREADS=4

echo "============================================================"
echo "[0/6] 数据预处理（跳过：data/real 已内置中间数据）"
echo "      如需重建: 见 README 4.0 (convert_real_data -> build_seq_tensors -> build_fusion_data)"
echo "============================================================"

echo ""
echo "============================================================"
echo "[1/6] 训练 Token-Attn 主模型（7 seed）"
echo "============================================================"
for s in 42 123 2024 7 99 500 2025; do
  echo "--- Token-Attn seed=$s ---"
  SEED=$s ONLY=tokenattn $PY scripts/train_c1c2.py
done

echo ""
echo "============================================================"
echo "[2/6] 训练对比基线（LightGBM / XGBoost / 端到端 Bi-LSTM）"
echo "============================================================"
$PY scripts/train_gbdt.py
$PY scripts/train_seq_bilstm.py

echo ""
echo "============================================================"
echo "[3/6] 7-seed 集成 + 统计显著性检验"
echo "============================================================"
$PY scripts/p0a_seed_ensemble.py
$PY scripts/p0b_stat_test.py

echo ""
echo "============================================================"
echo "[4/6] 可解释性归因（置换重要性 / 消融 / 注意力图 / 案例深描）"
echo "============================================================"
$PY scripts/r2_permutation_importance.py
$PY scripts/p0e_feature_ablation.py
$PY scripts/make_tokenattn_attn_figures.py
$PY scripts/r3_case_study.py

echo ""
echo "============================================================"
echo "[5/6] 成因机制分析（§4.5）"
echo "============================================================"
$PY scripts/r2_mechanism.py
$PY scripts/r2_causal_chain.py
$PY scripts/r2_mechanism_deep.py

echo ""
echo "============================================================"
echo "[6/6] 论文图 1 / 图 2"
echo "============================================================"
$PY scripts/fig1_fault_fingerprint.py
$PY scripts/fig2_roc_pr.py

echo ""
echo "============================================================"
echo "复现完成。关键输出："
echo "  docs/p0a_seed_ensemble.json      (Token-Attn 7-seed 集成 0.918)"
echo "  docs/p0b_stat_test.json          (bootstrap p=0.026 / DeLong p=0.321)"
echo "  docs/gbdt_compare.json           (LightGBM 0.868 / XGBoost 0.887)"
echo "  docs/routeC_bilstm_results.json  (Bi-LSTM 0.351)"
echo "  docs/r2_permutation_importance.json (置换重要性 Top5, 含 ±std)"
echo "  docs/p0e_feature_ablation.json   (消融)"
echo "  docs/r2_mechanism.json / r2_causal_chain.json / r2_mechanism_deep.json (§4.5)"
echo "  figures/fig1_fault_fingerprint.png / fig2_roc_pr.png"
echo "  docs/tokenattn_attn_figs/ (图3/4)  docs/r3_figs/ (图5)  docs/r2_figs/ (图6)"
echo "对照 README「论文核心结果一览」核对数值。"
echo "============================================================"
