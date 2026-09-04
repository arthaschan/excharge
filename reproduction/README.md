# 论文复现包（Reproduction Package）

> 论文：《基于可解释人工智能的充电站网络异常检测：面向欧盟《人工智能法案》合规方案》
> （TAIG 投稿，主模型 Token-Attn，纯 R2 路线）
>
> 本文件夹自包含复现论文全部数据所需的信息：环境初始化、数据来源、实验过程与代码。
> 按 `run_all.sh` 顺序执行即可从头复现论文的图 1–6、表 1–3 及正文所有关键数字。
> 已剔除过去失败/被否定的实验（TabPFN、数据增强、断点续训等过程性探索），只保留主线。

---

## 1. 论文核心结果一览（复现后对照用）

主指标为 **PR-AUC**（类别不平衡下的阈值无关排序指标），测试集为 owner7–8 全新站点
（2,776 序列，129 故障，序列级故障率 4.65%）。

| 指标 | 论文值 | 来源脚本 → 输出 |
|------|--------|----------------|
| Token-Attn（7-seed 集成）PR-AUC / AUC / F1 | 0.918 / 0.990 / 0.795 | p0a_seed_ensemble.py → p0a_seed_ensemble.json |
| Token-Attn 单 seed 均值±std | 0.874 ± 0.033 | 同上 |
| LightGBM PR-AUC / AUC | 0.868 / 0.987 | train_gbdt.py → gbdt_compare.json |
| XGBoost PR-AUC / AUC | 0.887 / 0.991 | 同上 |
| 端到端 Bi-LSTM PR-AUC / AUC | 0.351 / 0.908 | train_seq_bilstm.py → routeC_bilstm_results.json |
| SOTA 深度模型（iTransformer / PatchTST / FT-Transformer / DualTransformer）PR-AUC | 0.595 / 0.679 / 0.624 / 0.242 | train_newmodels.py → newmodels_*.json |
| 重建式 LSTM-AE（无监督）PR-AUC / AUC | 0.248 / 0.912 | train_recon.py → recon_results.json |
| vs LightGBM bootstrap p 值（PR-AUC 差） | 0.026（边际显著） | p0b_stat_test.py → p0b_stat_test.json |
| vs XGBoost bootstrap p 值 | 0.143（未达显著） | 同上 |
| AUC DeLong 检验 p 值 | 0.321（不显著） | 同上 |
| 置换重要性 Top5 ΔPR-AUC（均值±std） | t2_max 0.488±0.020 / t1_mean 0.426±0.013 / t2_last 0.243±0.028 / t2_mean 0.237±0.011 / t1_last 0.158±0.010 | r2_permutation_importance.py → r2_permutation_importance.json |
| 特征组消融（全量 / 去基础统计量 / 去分段差分 / 去温度变化 / 去电池类型 / 去超温） | 0.916 / 0.303(−0.61) / 0.758(−0.16) / 0.792(−0.12) / 0.820(−0.10) / 0.885(−0.03) | p0e_feature_ablation.py → p0e_feature_ablation.json |
| 故障机制分型 | 电气中断型 123(95.3%) / 超温型 6(4.7%) | r2_mechanism.py → r2_mechanism.json |
| 数据规模 | 30 站 / 155.6 万采样点 / 19,658 有效序列 | convert_real_data.py / build_fusion_data.py |

---

## 2. 环境初始化

- 硬件（实测环境）：NVIDIA H100 NVL 96GB / CUDA 12.9
- Python：3.11（无 conda，直接用 venv 的 `bin/python`）
- 实测关键版本：torch 2.6.0+cu124、numpy 2.4.6、pandas 3.0.5、scikit-learn 1.9.0、
  xgboost 3.2.0、lightgbm 4.7.0、pyarrow 25.0.1、matplotlib 3.11.1、scipy 1.17.1

```bash
# 1) 建虚拟环境
python3.11 -m venv .venv
source .venv/bin/activate

# 2) 装 torch（按 CUDA 版本，例 cu124）
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 3) 装其余依赖
pip install -r requirements.txt
```

全部脚本用 `DEVICE=cuda`（或自动选 cuda）跑 GPU；无 GPU 也可跑 CPU（仅更慢，结论不变）。

---

## 3. 数据来源

见 `data/README.md`（含原始数据集 DOI、下载方式、三个中间数据文件的说明与划分口径）。
要点：

- 原始数据：Yang 等人公开数据集，Mendeley `10.17632/c7gg94tmvz.3`，
  代码 Zenodo `10.5281/zenodo.17423221`，文件 `processed_data.xlsx`（8 sheet）。
- 本文件夹已内置 `data/real/` 三个中间文件，**无需重新下载原始 xlsx** 即可复现
  论文全部结果；如需从原始 xlsx 开始，先跑 `scripts/convert_real_data.py`。

---

## 4. 复现步骤（完整管线，按序执行）

以下命令均在**本文件夹根目录**下执行。可逐条手跑，也可直接 `bash run_all.sh`。

### 4.0 数据预处理（数据已内置时可跳过）

```bash
# 仅当需要从原始 xlsx 重建时才跑；需先把 processed_data.xlsx 放到 data/raw/
python scripts/convert_real_data.py      # → data/real/all_data.parquet
python scripts/build_seq_tensors.py      # → data/real/seq_tensors.pkl
python scripts/build_fusion_data.py      # → data/real/fusion_data.pkl
```

### 4.1 训练主模型 Token-Attn（7 个 seed）

```bash
for s in 42 123 2024 7 99 500 2025; do
  SEED=$s DEVICE=cuda ONLY=tokenattn python scripts/train_c1c2.py
done
# → docs/c1c2_tokenattn[_s{seed}]_results.json / _prob.npy / _model.pt
```

### 4.2 训练对比基线

```bash
python scripts/train_gbdt.py             # LightGBM / XGBoost → docs/gbdt_compare.json + *_prob.npy
python scripts/train_seq_bilstm.py       # 端到端 Bi-LSTM → docs/routeC_bilstm_results.json + _prob.npy
python scripts/train_newmodels.py        # SOTA 深度模型(iTransformer/PatchTST/FTTransformer/DualTransformer) → docs/newmodels_*.json
python scripts/train_recon.py            # 重建式 LSTM-AE(无监督) → docs/recon_results.json
```

### 4.3 集成与统计检验

```bash
python scripts/p0a_seed_ensemble.py      # 7-seed 集成 → docs/p0a_seed_ensemble.json + ensemble_prob.npy
python scripts/p0b_stat_test.py          # bootstrap + DeLong → docs/p0b_stat_test.json
```

### 4.4 可解释性归因

```bash
python scripts/r2_permutation_importance.py   # 置换重要性（n=3, seed=0, 含 std/CI）→ docs/r2_permutation_importance.json
DEVICE=cuda python scripts/p0e_feature_ablation.py  # 特征组消融（5 次重训）→ docs/p0e_feature_ablation.json
python scripts/make_tokenattn_attn_figures.py # 注意力图 3/4 → docs/tokenattn_attn_figs/
python scripts/r3_case_study.py               # 案例深描（图 5）→ docs/r3_figs/case_*.png
```

### 4.5 成因机制分析（§4.5）

```bash
python scripts/r2_mechanism.py           # 分型 + 终止方式 + 图 6 → docs/r2_mechanism.json + r2_figs/
python scripts/r2_causal_chain.py        # 时序因果链 → docs/r2_causal_chain.json
python scripts/r2_mechanism_deep.py      # 温度-时长相关（r≈0.13）→ docs/r2_mechanism_deep.json
```

### 4.6 论文图 1 / 图 2

```bash
python scripts/fig1_fault_fingerprint.py # 图 1 → figures/fig1_fault_fingerprint.png
python scripts/fig2_roc_pr.py            # 图 2（四模型 ROC+PR）→ figures/fig2_roc_pr.png
```

---

## 5. 论文图表 ↔ 脚本 ↔ 输出 映射表

| 论文内容 | 脚本 | 输出 |
|----------|------|------|
| 表 1 性能对比 | train_c1c2.py + p0a_seed_ensemble.py + train_gbdt.py + train_seq_bilstm.py + train_newmodels.py + train_recon.py | docs/p0a_seed_ensemble.json、gbdt_compare.json、routeC_bilstm_results.json、newmodels_*.json、recon_results.json |
| §4.2 统计显著性（p=0.026/0.143/0.321） | p0b_stat_test.py | docs/p0b_stat_test.json |
| §4.3 置换重要性 Top5（含 ±std） | r2_permutation_importance.py | docs/r2_permutation_importance.json |
| §4.3 特征组消融（敏感性） | p0e_feature_ablation.py | docs/p0e_feature_ablation.json |
| 图 1 故障指纹（均值口径） | fig1_fault_fingerprint.py | figures/fig1_fault_fingerprint.png |
| 图 2 ROC/PR 曲线（四模型） | fig2_roc_pr.py | figures/fig2_roc_pr.png |
| 图 3/4 自注意力热力图/偏移 | make_tokenattn_attn_figures.py | docs/tokenattn_attn_figs/ |
| 图 5 案例深描（4 样本） | r3_case_study.py | docs/r3_figs/case_*.png |
| 图 6 机制分型/终止方式/温度角色 | r2_mechanism.py | docs/r2_figs/mechanism_subtypes_causal.png |
| §4.5 时序因果链（前兆→中断→后果） | r2_causal_chain.py | docs/r2_causal_chain.json |
| §4.5 温度-时长弱相关 r≈0.13 | r2_mechanism_deep.py | docs/r2_mechanism_deep.json |
| §3.1 数据规模/电池类型/划分 | convert_real_data.py + build_fusion_data.py | data/real/ 与脚本打印 |

---

## 6. 复现注意事项（重要）

1. **随机种子与确定性**
   - 数据划分 seed=42 分层抽样；Token-Attn 训练 seed 见 4.1（42/123/2024/7/99/500/2025）；
   - 置换重要性 `np.random.default_rng(0)`，n=3 次置换取平均，完全可复现；
   - GBDT/XGBoost `random_state=42`。

2. **硬件浮点差异 + batch size（重训 vs 原结果）**
   - 论文数字来自原环境（Apple Silicon MPS）的运行结果。
   - **⚠️ batch size 必须为 64**：`train_c1c2.py` 原版用 `BATCH=64`；若误用默认大 batch（如 256），
     Token-Attn 单 seed 会从 ~0.87 暴跌到 ~0.60（小样本强不平衡下优化动态显著变差）。
     本仓库已把默认改回 `BATCH=64`（2026-09-04 修复），复现 0.918 务必确认 `BATCH=64`。
   - 在 H100（CUDA）上以 `BATCH=64` 重训，单 seed 42 实测 PR-AUC≈0.916，与论文 0.918 吻合；
     cuDNN 浮点舍入带来小幅波动（单 seed 标准差本就约 ±0.033），结论不变。
   - **加载已保存 checkpoint 重跑推理/归因**（非重训）可逐位复现到小数点后 4 位，
     例如置换重要性 base PR-AUC=0.91576975352594 与原始结果一致到 15 位有效数字。

3. **均值 vs 中位数口径**
   - §4.3 故障指纹表为**均值**（末端 SOC 78.7%/94.3%，时长 31.2/56.4 min）；
   - §4.5 机制分析为**中位数**（末端 SOC 84%/98%，时长 25/52 min，枪温 38/48°C）。
   - 两者是同一指标的不同统计量，非矛盾；fig1 用均值，r2_mechanism 用中位数。

4. **序列截断**：MAXLEN=200，超长序列取前 200 步（占比 <2%，对故障 <1%），
   62 维特征在完整序列上计算，不受截断影响。

5. **主指标**：PR-AUC（阈值无关）；F1 为固定阈值 th=0.5 工作点；AUC 被多数类主导
   对少数类不敏感，故统计显著性的结论以 PR-AUC 的 bootstrap 为准。

6. **数据划分口径不可变**：owner1–6 训练（内部 80/20 切 val）、owner7–8 测试。

---

## 7. 与论文同步约定

本复现包覆盖论文的**实验性数据**（图 1–6、表 1–3、正文所有关键数值），
这些数值必须与 `paper/paper_draft.md` 保持一致；本 README「核心结果一览」表与
`data/README.md`「数据划分口径」即为对照基准。

- **需同步**：论文实验性数值（性能、显著性 p 值、置换重要性、消融、分型、
  数据规模/故障率/电池类型分布等）一旦改动 → 同步更新本 README 的表、data/README
  的统计值，以及对应脚本/输出。
- **无需同步**（仅 paper/ 维护）：参考文献条目（年份/文章号）、作者列表、
  非实验性行文措辞（如"达到并部分超过 GBDT"）。
