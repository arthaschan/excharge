# journal/code/ —— 期刊代码

> 期刊论文的实验代码放这里。命名与纪律沿用 `earlywarning/code/`（见 `journal/README.md` 第三节）。

## 规划中的脚本（P0–P4，尚未开跑）

| 优先级 | 计划脚本 | 用途 | 状态 |
|--------|---------|------|------|
| P0 | `p0_baseline_table.py` | 钉靶：全序列/前缀基线对照表 | 待写 |
| P1 | `p1_explanation_features.py`（+ 改 `build_prefix_features.py` 配方） | 6~8 个反哺物理特征 + LightGBM/TokenAttn paired 对照 | 待写 |
| P2 | `p2_attribution_regularization.py` | 归因一致性正则 / 辅助头（仅 TokenAttn） | 待写 |
| P3 | `p3_feature_selfsupervised.py` | 特征级自监督预训练 | 待写 |
| P4 | `p4_sota_baselines.py` | SOTA 基线补齐（Anomaly Transformer/TranAD/USAD/GDN/DeepSVDD/DCdetector） | 待写 |

> 复用现有代码：`earlywarning/code/build_prefix_dataset.py`、`build_prefix_features.py`、`build_prefix_seq_tensors.py`、`train_prefix_tokenattn.py`、`extract_prefix_features.py`、`ensemble_eval.py`。
