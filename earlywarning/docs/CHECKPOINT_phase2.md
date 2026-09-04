# CHECKPOINT — earlywarning 进度快照(Phase 3 完成态 · 暂停点)

> **状态:Phase 3 完成(2026-09-04)并正式暂停。** 方案 C 已定型(树为主报 + TokenAttn 副线);Phase 3a EAR / 3b 双谱系 / 3c 注意力 / 3d 样例图 全部产出,可直接进入 Phase 4 论文初稿。  
> **暂停原因**: 用户已转向老师给的"另外一个期刊方向",明确要求"本阶段完成后暂停后续任务,过程已记录,日后某一天重启"。

---

## 0. 暂停点快照(2026-09-04,重启第一站)

**重启时只需读本节 + §7 重启指引,即可完整恢复上下文。**

- ✅ **Phase 3 已 100% 收尾,无任何后台任务在运行**: 训练日志全部止于 09:25(`tune_d128.log`);Phase 3 四个分析 09:29–09:38 全部跑完(`phase3_ear.py` / `phase3b_lineage.py` / `phase3c_attn.py` / `phase3d_viz.py`),产物齐全。
- 📄 **已交付文档(本次暂停的完整过程记录)**:
  1. `docs/gate_report_phase3.md` — Phase 3 总门控报告(数据口径/EAR/机制/注意力/样例/边界/Phase 4 任务)
  2. 本文件 `docs/CHECKPOINT_phase2.md` — 里程碑 + 重启命令 + 经验教训
  3. `docs/研究方案_充电过程在线早期预警_v1.0.md` — 总纲
- 🔜 **唯一待重启任务 = Phase 4 论文初稿**(指引见 `gate_report_phase3.md` §8 的 P4-1~P4-6)。
- 🗂️ 早期任务 `#16/#17`(会议版 tokenattn checkpoint + 注意力图)在更早已被取消(metadata `cancelled:true`),其目标已被 Phase 3c/3d 以新方向实现覆盖,**不再重启**。

---

## 1. 项目定位(30 秒版)

**充电过程在线早期预警(双谱系)** — 把会议版"事后判断整条记录异常"升级为"充电进行中只许看当前前缀 `[begin, begin+τ]`,预测最终会否异常终止",分启动型(<30min 即断,占故障 85%)与运行型(≥30min)分别建模。会议保底 · 期刊冲刺。历史经验纪律见《模型选型判定手册》(R0-R12 / E1-E12,核心:E11 前缀特征禁偷看未来、跨域训练禁混洗、PR-AUC 主指标、R6 深度=多seed集成+统计检验、R9 探路→量产)。

## 2. 已完成里程碑(全部可复现)

| 阶段 | 内容 | 结论/产物 |
|---|---|---|
| Phase 0 | 决策门 | 4/4 全绿(τ=1 PR-AUC 0.877 等)→ `gate_report_phase0.md` |
| Phase 1 | 数据管线冻结 + 结构 A/B | 选 **A 两阶段级联** → `gate_report_phase1.md` |
| Phase 2 探路 | Token-Attn 1-seed 探路 | 5 τ 探路完成 → `gate_report_phase2_probe.md` |
| Phase 2 方案 A | 7-seed 集成试点 | 深度追平未反超树(Δ=−0.029, p=0.07~0.15)→ `gate_report_phase2.md` |
| Phase 2 方案 B | τ=3 调参网格诊断 | 7 变体均 ≤0.871,**欠调参假说排除**,转 C |
| Phase 2 方案 C | 树为主+深度副线 | 落地:LightGBM 主报,TokenAttn 作 Phase 3 可解释载体 |
| **Phase 3a** | **EAR(最早可预警前缀)分析** | **完成** → startup 65.9 % / run 81.5 % 预警率,中位 lead 5.6 / 37.6 min |
| **Phase 3b** | **双谱系机制对比** | **完成** → LightGBM top 重要性 + τ=2 三族通道级画像 |
| **Phase 3c** | **TokenAttn 注意力归因** | **完成** → CLS gunT2 聚焦(双族 0.47~0.49 vs normal 0.24) |
| **Phase 3d** | **样例可视化** | **完成** → `fig_ear_case_{startup,run,normal}.png` 3 张 |
| **Phase 3 门控** | **总报告** | **`gate_report_phase3.md`(本次产出)** |

## 3. 方案 C 最终性能(深度 vs 树,Phase 2 裁决后定性)

### 7-seed 集成 vs LightGBM 同协议

| τ | 深度 7-seed | LightGBM | Δ | P(深度>树) | 判定 |
|---|---|---|---|---|---|
| 3min | 0.8944 | **0.9238** | −0.029 | 0.070 | 无显著差异(接近) |
| 5min | 0.8664 | **0.8952** | −0.029 | 0.145 | 无显著差异 |

**结论**: 深度没有输(不显著),也没证明赢;点估计稳定落后 ~0.03。**结构性结论**——前缀任务表侧特征携带主要信号,树竞争力更强,序列增量贡献被压缩;非欠调参、非训练 bug。Phase 2 方案 B 7 变体(lr3e-4/3e-3、k4、l1、d128)全部 ≤ 0.871,坐实默认配置已位于局部最优。

单 seed 波动 ±0.06(τ=3 范围 0.792~0.908),印证 R6 纪律。

### Phase 3 EAR 汇总(主报 LightGBM)

| 族 | n | 预警率 | EAR 中位 | lead_min 中位 | 未预警 |
|---|---|---|---|---|---|
| startup | 317 | 65.9 % | 2.0 min | **5.6 min** | 108 (34.1 %) |
| run | 54 | 81.5 % | 2.0 min | **37.6 min** | 10 (18.5 %) |

### 决策链(用户拍板轨迹)

1. 探路后 → 用户选 **"先 A 后 B"**  
2. 方案 A 完成 → 用户再选 **"先 B 后 C"**  
3. 方案 B 完成(2026-09-04 09:25) → **裁定转 C**(2026-09-04 09:30)  
4. Phase 3 完成(2026-09-04) → **待用户重启后进 Phase 4 论文**

## 4. 当前文件状态

### 关键结论文档(docs/)

- `gate_report_phase0.md` / `gate_report_phase1.md` / `gate_report_phase2_probe.md` / `gate_report_phase2.md`  
- **`gate_report_phase3.md`(本次新交付,Phase 3 总览)** ← 重启必读  
- `structure_ab_results.json` / `ensemble7_tau3_results.json` / `ensemble7_tau5_results.json`  
- `phase3_ear_results.json` / `phase3_ear_by_txn.csv` / `phase3b_lineage.json` / `phase3c_attn_results.json`  
- `fig_ear_case_{startup,run,normal}.png`(3 张样例图,论文素材)  
- `研究方案_充电过程在线早期预警_v1.0.md`(总方案)

### 数据(data/,Phase 1 冻结,勿改)

- `prefix_dataset_full.parquet`(7,386 段前缀,2,941 事务) · `split.json` · `prefix_feats_v1.parquet`(52 维) · `seq_tensors_tau{1,2,3,5,10,20}.pkl`  
- ⚠️ **已知数据语义**: 特征列 `power_peak_pos` 在"前缀内功率全 0"时为 nan。LightGBM 原生处理;深度模型训练脚本已做 E11 安全中位数填充。**τ=1 cohort 仅 3,524("首分钟 ≥2 行"人群窄)**,Phase 0 τ=1 数字仅代表该子人群。  
- ⚠️ **offsets 语义** = 自"插枪"计时(非起充)。部分事务因延迟起充,τ=1/2/3 前缀行 <2 → 不可用 τ 不计入 EAR。论文 §诚实边界 必须说明。

### 代码(code/)

- 数据管线(Phase 1 冻结):`build_prefix_dataset.py` / `build_prefix_features.py` / `build_prefix_seq_tensors.py` / `structure_ab.py`  
- 模型:`train_prefix_tokenattn.py`(支持 env 超参覆盖,`LR/K_SEG/N_LAYERS/D_MODEL/N_HEADS/FF/DROPOUT/TAG`)、`ensemble_eval.py`  
- Phase 2 调参:`run_tuneB.sh` / `summarize_tuneB.py`(已完成,产物 docs/prefix_tokenattn_tau3_tune_*.json)  
- **Phase 3(本次交付)**:`phase3_ear.py` / `phase3b_lineage.py` / `phase3c_attn.py` / **`phase3d_viz.py`(已修复 32 位 tid 解析 bug)**  
- 模型权重(`docs/prefix_tokenattn_tau3_s{0..6}_*.pt` 等):Phase 3 可重复加载,无需重训

## 5. Phase 3 关键产物(摘要)

### 5.1 Phase 3a — EAR

- 故障事务 371(startup 317 / run 54),正常段 2,284  
- 逐 τ 高精度阈值:τ1=0.998 / τ2=0.802 / τ3=0.832 / τ5=0.806 / τ10=0.769 / τ20=0.639  
- EAR 分布(startup vs run):τ=2 占绝对多数(144/23)  
- 输出: `phase3_ear_results.json` + `phase3_ear_by_txn.csv`(1,314 行,含每事务全 τ 概率轨迹)

### 5.2 Phase 3b — 双谱系机制(LightGBM 视角)

- Stage2 top 重要性: chargingv_slope(655) / SOC_last(625) / SOC_mean(549) / chargingv_first(537) / chargingv_last(469)  
- τ=2 三族画像核心差异:  
  - **startup** = 短促大电流主动注入型(charginga_first 114.7 A vs normal 60.9,power_active_ratio 0.994)  
  - **run** = 长时间高压缓充型(chargingv_first 357.9 V,SOC 起步 10.5 % 更低平缓,power_active_ratio 1.000)  
- 输出: `phase3b_lineage.json`

### 5.3 Phase 3c — TokenAttn 注意力归因(副线)

- **CLS 通道级 gunT2 高度聚焦**: startup 0.475 / run 0.491 vs normal 0.244 → 故障族 gunT2 通道权重 2x normal  
- **段级注意力差异**: normal 集中前 2-3 段(0.12/0.14/0.09),startup/run 均匀分散(各 0.01~0.02) → 故障族需看更长序列  
- **特征级聚焦**: indices 8-12(chargingv/charginga/out_power first/last/mean),与 LightGBM top 重要性特征组重合  
- 输出: `phase3c_attn_results.json`(τ=3 seed0)

### 5.4 Phase 3d — 样例可视化

- 3 张 PNG 已生成,标签与 EAR 数据驱动一致,论文可直接使用  
- 修复记录(本次):CSV 32 位 transaction_id 被 pandas 解析为 object(int),与 str 目标 `==` 恒 False → 已修复 `dtype={'transaction_id': str}`;EAR/lead 改为数据驱动;normal 不画误导 EAR 竖线;坐标轴标签由"time since session start"修正为"time since plug-in (min)"

## 6. 下一步(Phase 4 论文初稿,待重启)

详见 `gate_report_phase3.md` §8。

| # | 任务 | 锚定产物 |
|---|---|---|
| P4-1 | 论文骨架: title + abstract + intro + method + experiments + conclusion | `paper/` |
| P4-2 | Method 节核心表: 双族机制画像 + EAR 提前量 + 主副线互补表 | `gate_report_phase3.md` §3.2 / §4.1 / §6 |
| P4-3 | 实验节: Phase 2 主报结果 + Phase 3 EAR 汇总 + 注意力热图 | `phase3_ear_results.json` / `phase3c_attn_results.json` |
| P4-4 | 论文核心图(4 张): (a) EAR 累积分布 (b) 段级注意力对比 (c) 通道级注意力对比 (d) 3 张样例图 | `fig_ear_case_*.png` |
| P4-5 | 诚实边界段: `gate_report_phase3.md` §7 1~6 条作为论文 §4 limitation | |
| P4-6 | 期刊扩展清单: 多站点验证 / 滑动窗口在线部署 / 数据延迟容忍 | 后续论文扩展 |

## 7. 重启指引(快速恢复状态)

```bash
cd /Users/arthas/git/excharge/earlywarning
PY=/Users/arthas/.workbuddy/binaries/python/envs/default/bin/python

# ① 必读三份文档(按顺序)
#    docs/CHECKPOINT_phase2.md (本文件,Phase 2→3 进度快照)
#    docs/gate_report_phase3.md (Phase 3 总览)
#    docs/研究方案_充电过程在线早期预警_v1.0.md (总纲)

# ② 验证产物齐全
ls -la docs/phase3_*.{json,csv} docs/fig_ear_case_*.png

# ③ 可选: 重生成样例图(若未来 CSV 改了或模型权重需重跑)
$PY code/phase3d_viz.py

# ④ 启动 Phase 4 论文
# 见 §6 P4-1 ~ P4-6
```

## 8. 本方向关键经验教训(重启必读,避免重复踩坑)

1. **loss=nan 先查特征数据缺失值** — `power_peak_pos` 的 nan → z-score 整列污染 → 前向 nan。曾误诊为"MPS 概率性 backward bug",CPU 复现 + 逐列 isna 扫描才锁定真凶。LightGBM 不暴露此类问题,深度模型必显式填充(训练域中位数,E11 安全)。  
2. **本机 Edit 工具多次"报成功但未落盘"** — 任何 Edit 后必须用 Read/grep/python 验证落盘再继续。  
3. **训练输出勿覆盖** — MPS 4-epoch 验证曾用同名输出覆盖 τ=5 的 30-epoch 正确结果(0.857→0.117)。验证性短跑必须设不同 TAG/SEED/输出名。  
4. **CPU 并行勿超线程预算** — 2 进程并行(各 4 线程)共享 10 核,14 训练耗 7h;串行单进程反而更快。  
5. **grep 在此环境呈 ripgrep 风格**(`\|` 按字面处理)— 用 Grep 工具或 python 字符串计数验证。  
6. **τ=1 cohort 边界** — ~1min/采样点,"首分钟 ≥2 行"仅占 dur≥1 人群 12 %;论文写 EAR 必须写清预测人群定义。  
7. **CSV 32 位 ID 解析陷阱(本次新增)** — pandas read_csv 将 32 位 transaction_id 解析为 object(Python int)而非 str,与 str 目标 `==` 恒 False → 读取时显式 `dtype={'transaction_id': str}`。parquet 因类型已定,不受影响。  
8. **TransformerEncoderLayer 注意力抓取(本次新增)** — torch≥2.x 默认 `need_weights=False`,output[1]=None;hook 内重调用 `need_weights=True` 会递归。手写前向逐层复刻(每层 norm→self_attn(need_weights=True)→残差→ff_block)拿权重。  
9. **import 训练脚本会触发模块级数据加载** — `from train_prefix_tokenattn import ...` 会执行 `pd.read_parquet(...)` 等撑爆内存 → 内嵌必要类定义(本次 TokenAttnPrefix + BiLSTMBackbone + make_len_mask),避免 import。