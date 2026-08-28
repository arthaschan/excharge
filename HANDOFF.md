# HANDOFF —— excharge 项目交接文档（给 workbuddy）

> 更新：2026-08-28 12:55 ｜ 交接人：高级项目经理 Agent ｜ 接手人：workbuddy
> 项目仓库：`/Users/arthas/git/excharge`（git@github.com:arthaschan/excharge.git，main 分支）

---

## 一、项目一句话

**可解释充电站异常检测论文**（投 TAIG 会议，主题 = AI 治理技术）。
基于深圳 30 座真实充电站数据（155.6 万采样点），做故障异常检测 + 可解释归因 + EU AI Act 合规框架。核心卖点不是"检测性能最高"，而是"**可解释、可审计、符合监管**"。

**论文作者**（务必用对，之前踩过坑）：
- 第一作者：王莹（珠海学院 应用人工智能理学硕士）
- 第二作者：陈天元（用户本人，同专业硕士）
- 通讯作者：朱禹林（珠海学院教授）

---

## 二、核心资产位置

| 资产 | 路径 | 说明 |
|------|------|------|
| 代码仓库 | `/Users/arthas/git/excharge` | git 已推到 origin/main |
| 真实数据 | `data/real/all_data.parquet` | 155.6 万行，34.6MB，**未入库**（本地） |
| 原始数据包 | 项目根目录 `A dataset of EV batery...zip`（207MB） | **未入库**，本地 |
| 论文正文 | `paper/paper_draft.md` | 当前主线（Bi-LSTM + GradCAM） |
| 论文备份 | `paper/paper_draft_xgb_backup.md` | XGBoost 版（历史留档） |
| 参考文献 PDF | `references/`（16 篇） | 本地留存，**不入库**（gitignore） |
| 图 | `paper/figures/`（8 张） | 已入库 |
| 脚本 | `scripts/`（26 个） | 已入库 |

**git 状态**：工作区干净（仅 `paper/paper_draft.pdfbuild.md` 是导出 PDF 的临时副本，未入库，可删可留）。
最新 commit `6d15e5e` 已推送 origin/main。分支 `main`。

---

## 三、方法路线演进史（关键！接手必读）

| 阶段 | 方法 | 核心指标 | 状态 |
|------|------|---------|------|
| ① 早期 | XGBoost + 57 维手工特征 + TreeSHAP | AUC=0.988，PR-AUC=0.893，Recall=86.05%(th=0.075)，**超 Nature 基线 73.56%** | 性能最好，但被导师否 |
| ② 当前 | 轻量 Bi-LSTM（6 通道原始时序）+ 1D-GradCAM | AUC=0.908，**PR-AUC=0.351**，Recall=77.52%(th=0.5) | 符合导师要求，但性能暴跌 |
| ③ 新建议 | **深度模型 + 手工特征融合**（Bi-LSTM 128维序列表示 + 57维特征拼接） | 目标：PR-AUC 拉回 0.86~0.89 | **未实施，待决策** |

**Nature 基线论文**（Yang H, et al., Nat Commun 17:974, 2026）：个性化联邦 Transformer，同分布评估 Recall=73.56%。本文是跨站点设定，不直接可比（已在论文中如实标注）。

---

## 四、⚠️ 当前待决问题（最高优先级）

### 问题 1：性能 trade-off（核心矛盾）

导师 5 条批注要求换深度模型（XGBoost→Bi-LSTM），导致性能暴跌：

| 指标 | XGBoost(旧) | Bi-LSTM(新) | 变化 |
|------|------------|-------------|------|
| AUC | 0.988 | 0.908 | ↓0.080 |
| PR-AUC | 0.893 | 0.351 | ↓0.542 |
| F1(th=0.5) | 0.838 | 0.359 | ↓0.479 |
| Recall | 86.05% | 77.52% | ↓8.5pp |

**风险**：PR-AUC 0.351 在"高风险系统"场景下误报率高，投稿大概率被质疑。

### 问题 2：新方案是否采纳（见根目录 docx）

`TAIG_论文方法路线建议.docx`（2026-08-28 12:51 新增）提出**折中方案**：
- 保留 Bi-LSTM 做时序编码（128维序列表示）
- 把 57 维手工特征拼回去（而非端到端只喂 6 通道）
- 可解释性双轨：1D-GradCAM（时序）+ SHAP（特征）
- 预期 PR-AUC 拉回 0.86~0.89，同时满足导师批注 1/2

**这是交接给 workbuddy 需要首先和用户确认的核心决策**：走纯 Bi-LSTM（性能差但简单）还是融合方案（性能好但复杂）？

---

## 五、导师 5 条批注（2026-08-24 提出）

| # | 批注原文 | 落实状态 |
|---|---------|---------|
| 1 | XGBoost 老旧，换深度时序模型（RNN/LSTM/Transformer） | ✅ 换 Bi-LSTM |
| 2 | SHAP 经典，换神经网络专用方法（如 GradCAM） | ✅ 换 1D-GradCAM |
| 3 | "特征消融"→"敏感性分析" | ✅ §4.5 逐通道置零 |
| 4 | 联邦学习与主题不符 | ✅ 删，改"站点自适应参数生成"，标注不直接可比 |
| 5 | "时序动力学特征"需说明提取方法 | ✅ 变相落实（删手工特征，端到端） |

详见根目录 `TAIG_论文修改问题总结.docx`（含完整核对表 + 遗留硬伤清单）。

---

## 六、数据口径（写论文/实验务必一致）

- **数据集**：深圳 30 座公共充电站，2020-10 ~ 2023-10
  - 全量 `processed_data.xlsx`：155.6 万采样点、31,449 序列
  - >30 分钟子集 `processed_data_longer_than_30.xlsx`：154.7 万采样点（**本文未用**）
  - 数据 DOI（Mendeley）：10.17632/c7gg94tmvz.3；代码 DOI（Zenodo）：10.5281/zenodo.17423221
- **划分**：owner 1-6 训练（13,505 序列/642 故障，内部 80/20 切 Val）、owner 7-8 测试（2,776 序列/129 故障）
  - ⚠️ 注意：旧 XGBoost 用 owner 1-4 训练（12,484 序列），口径不同，表 1 已标注
- **输入**：6 通道原始时序 = 电压(chargingv)/电流(charginga)/功率(out_power)/枪温1/枪温2/SOC，z-score 归一化，padding 至 200 步
- **故障率**：采样点级 5.68%，序列级 4.74%（≥30 采样点筛选出 19,658 序列）

---

## 七、论文现状与已知遗留问题

**论文当前**：`paper/paper_draft.md`（346 行，中文），Bi-LSTM + GradCAM 主线，16 篇参考文献全部核实真实（已修正 [4] Isolation Forest 作者、[16] TimeVQVAE-AD 作者+标题）。

**图编号跳号（未修复）**：
- 图 1→图 2→**图 5**→图 6→图 7→图 8（缺图 3、图 4）
- 两个"表 4"（第 214 行 + 第 250 行，后者应为表 5）

**PDF 导出**：
- 已生成 `paper/论文_可解释充电站异常检测.pdf`（14 页，正常）
- 导出命令：pandoc + xelatex + PingFang SC，emoji 需先删（临时副本 `paper_draft.pdfbuild.md`）

---

## 八、参考文献（16 篇，已核实）

全部 PDF 已下载到 `references/`（本地，不入库）。关键修正记录：
- [4] Isolation Forest → 作者 Liu FT, Ting KM, Zhou ZH（原误写 Liu R）
- [16] TimeVQVAE-AD → 作者 Lee D, Malacarne S, Aune E，标题 "Explainable time series anomaly detection using masked latent generative modeling"（原误写 Mannion T）
- [3] Nature 论文 → 一作 Haosen Yang（原误写 KC B，张冠李戴）
- 其余 13 篇（USAD/GDN/Anomaly Transformer/OmniAnomaly/Attention is not Explanation/TreeSHAP/SHAP/Model Cards/EU AI Act/Grad-CAM/Integrated Gradients/MTAD-GAT/Ismail Fawaz）均核实正确

---

## 九、下一步建议（供 workbuddy 参考）

1. **【阻塞点，先问用户】** 性能 trade-off：纯 Bi-LSTM vs 深度+手工特征融合方案（根目录 2 个 docx 是刚出的关键材料）
2. 修复图/表编号跳号（图5→3、图6→4、图7→5、图8→6；第二个表4→表5）
3. 英文版 TAIG 投稿翻译（同步所有修正，仍挂起）
4. 若采纳融合方案 → 重跑实验 + 重写方法章节 + 重新出图

---

## 十、环境备忘

- Python 环境：pandas 3.0.5、numpy 2.2.6、torch 2.13.0（**MPS 可用**，别再用 CPU 白等）、xgboost 3.2.0、shap 0.51.0、sklearn 1.9.0、matplotlib 3.11.0
- pandoc 3.9 + xelatex + ctex（导出 PDF 用）
- 大文件一律不入 git（parquet/csv/pdf/docx/zip 已在 .gitignore）
- 训练/归因/出图的完整链路脚本都在 `scripts/`，可复现
