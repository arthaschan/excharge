# journal_earlywarning/ —— 期刊论文方向 ②（充电过程在线早期预警）

> **这是第 2 个期刊论文研究文件夹**，与 `journal/`（方向①：可解释性反哺检测，已证伪）平行独立。
> 本文件夹研究的是**被暂停后重启**的方向：充电过程在线早期预警（line1）。

---

## 一、定位（30 秒版）

把会议版「事后判断整条记录是否异常」升级为「**充电进行中，只许看当前前缀 `[begin, begin+τ]`，预测这次充电最终会否异常终止**」，并把故障分成**启动型（<30 min 即断，占 85%）**与**运行型（≥30 min 中断，占 15%）**双谱系分别建模，给出各自**最早可预警时刻（EAR）**与**跨数据方冷启动**证据。

## 二、为什么暂停、为什么重启

- **暂停点**：Phase 0-3 已 100% 完成（数据管线冻结 + 方案 C 定型「树为主报 + Token-Attn 副线」+ EAR/双谱系/注意力/样例图全部产出），停在 **Phase 4 论文初稿**。
- **暂停原因**：用户转向老师给的「可解释性反哺」方向（→ `journal/`，本日已系统验证为负并收口）。
- **重启**：现按 `earlywarning/docs/gate_report_phase3.md` §8 的 P4-1~P4-6 推进论文初稿。

## 三、资产（复用，勿重造）

| 资产 | 位置 | 状态 |
|------|------|------|
| 研究方案总纲 | `earlywarning/docs/研究方案_充电过程在线早期预警_v1.0.md` | ✅ 冻结 |
| Phase 3 总门控报告（含 §8 Phase 4 指引） | `earlywarning/docs/gate_report_phase3.md` | ✅ 冻结 |
| 进度快照（暂停点） | `earlywarning/docs/CHECKPOINT_phase2.md` | ✅ 冻结 |
| 数据（已重建） | `earlywarning/data/prefix_dataset_full.parquet`（31,449 事务）、`prefix_feats_v1.parquet`（52 维） | ✅ 冻结，勿改 |
| EAR 结果 | `earlywarning/docs/phase3_ear_results.json` / `phase3_ear_by_txn.csv` | ✅ |
| 双谱系机制 | `earlywarning/docs/phase3b_lineage.json` | ✅ |
| 注意力归因 | `earlywarning/docs/phase3c_attn_results.json` | ✅ |
| 样例图 | `earlywarning/docs/fig_ear_case_{startup,run,normal}.png` | ✅ |
| 会议版论文（参照） | `paper/paper_draft.md` | ✅ |

## 四、Phase 4 计划（P4-1 ~ P4-6）

| # | 任务 | 锚定产物 | 状态 |
|---|---|---|---|
| P4-1 | 论文骨架：title + abstract + intro + method + experiments + conclusion | 本夹 `paper_draft.md` | 🔄 已起骨架 |
| P4-2 | Method 节核心表：双族机制画像 + EAR 提前量 + 主副线互补表 | gate_report_phase3 §3.2/§4.1/§6 | 待写 |
| P4-3 | 实验节：Phase 2 主报 LightGBM + Phase 3 EAR 汇总 + 注意力热图 | phase3_ear_results.json / phase3c_attn_results.json | 待写 |
| P4-4 | 论文核心图 4 张：(a) EAR 累积分布 (b) 段级注意力 (c) 通道级注意力 (d) 3 张样例图 | fig_ear_case_*.png | 待出图 |
| P4-5 | 诚实边界段（§7 六条 → 论文 limitation） | gate_report_phase3 §7 | 待写 |
| P4-6 | 期刊扩展清单：多站点验证 / 滑动窗口在线部署 / 数据延迟容忍 | — | 待列 |

## 五、关键数字（勿再改错，全部可复现）

- **数据**：31,449 事务，5,680 故障（18.06%）；启动型 4,827（85%，其中 <5min 3,610）/ 运行型 853；8 数据方，Sheet7 故障率 63.5% vs Sheet8 1.4% 极端反差。
- **检测质量（会议版口径）**：Token-Attn 7-seed 0.918 / XGBoost 0.887 / LightGBM 0.868 / Bi-LSTM 0.351。
- **EAR（test owner7-8，故障 371 = startup 317 + run 54）**：
  - startup 预警率 **65.9%**（209/317），EAR 中位 2min，lead 中位 **5.6min**；未预警 108（34.1%）。
  - run 预警率 **81.5%**（44/54），EAR 中位 2min，lead 中位 **37.6min**；未预警 10（18.5%）。
  - 逐 τ 高精度阈值：τ1=0.998 / τ2=0.802 / τ3=0.832 / τ5=0.806 / τ10=0.769 / τ20=0.639。
- **双谱系机制（LightGBM）**：startup = 短促大电流主动注入（charginga_first 114.7A vs 60.9，power_active_ratio 0.994）；run = 长时间高压缓充（chargingv_first 357.9V vs 340.2，SOC 起步 10.5% 更低）。
- **注意力归因（Token-Attn）**：CLS 通道级 gunT2 高度聚焦（startup 0.475 / run 0.491 vs normal 0.244）；normal 段级注意力集中前 2-3 段，故障族均匀分散。
- **7-seed 集成 vs LightGBM（前缀）**：τ=3 0.894 vs 0.924（Δ−0.029）、τ=5 0.866 vs 0.895（Δ−0.029）——深度不输不赢，树为主报。

## 六、诚实边界（论文 limitation 素材）

1. <5min 启动型故障物理预警窗口极窄（只能用 1-4min 前缀）；启动型建模为「插枪后头几分钟极短前缀判别」，运行型才是 5-30min 前缀主场。
2. EAR/lead 在 owner1-6 切分上得出，未在外部站点独立验证（P4-6 多站点交叉验证）。
3. offsets 语义 = 自「插枪」计时，非起充；部分事务延迟起充 → τ=1/2/3 前缀行 <2 不可用，不计入 EAR。
4. τ=1 cohort 仅「首分钟 ≥2 行」子人群（3,524），τ=1 数字仅代表该子人群。
5. `power_peak_pos` 在「前缀内功率全 0」时为 nan（深度模型已做训练域中位数填充）。
6. 无细粒度故障原因标签（仅二值 label），机制为数据支撑假设。

## 七、下一步

从 P4-1 骨架起步 → P4-2/P4-3 填表 → P4-4 出图 → P4-5/P4-6。每完成一项在 `reports/` 落一篇汇报（沿用 journal/ 的「每次汇报一个文档」约定）。

*本文件夹由 2026-09-04 会话创建，重启早期预警期刊方向。*
