# excharge — Explainable Anomaly Detection for EV Charging Stations

面向充电站网络的可解释异常检测系统，结合时序深度学习和 GradCAM 可解释性分析。

> **投稿目标**：TAIG Workshop / International Workshop on Technologies for AI Governance

## 项目结构

```
excharge/
├── README.md                  # 项目说明
├── .gitignore                 # 忽略规则
├── data/                      # 数据目录（gitignore）
│   ├── stations.csv           # 充电站清单
│   ├── piles.csv              # 充电桩清单
│   ├── fault_records.csv      # 故障标注
│   └── processed/             # 预处理后数据
├── scripts/                   # 代码
│   ├── generate.py            # 模拟数据生成
│   ├── preprocess.py          # 特征工程 + 标注
│   ├── train.py               # 模型训练
│   └── explain.py             # GradCAM 可解释性
├── docs/                      # 文档
│   ├── RESEARCH_PLAN.md       # 研究计划
│   ├── literature_review.md   # 文献调研
│   └── PREPROCESSING_REPORT.md# 预处理报告
└── results/                   # 训练结果
    └── model_comparison.csv
```

## 数据规模

- 100 个充电站 × 20 个充电桩 = 2,000 桩
- 365 天 × 15 分钟粒度 = 126 万条时序记录
- 6 种故障类型，故障率 ~1%

## 技术栈

- **模型**：LSTM / Transformer Encoder + Focal Loss
- **可解释性**：1D-GradCAM + SHAP
- **评估**：F1 / AUC-ROC + Fidelity / AOPC

## 作者

- 第一作者：王莹（珠海学院应用人工智能理学硕士）
- 第二作者：陈天元（珠海学院应用人工智能理学硕士）
- 通讯作者：朱禹林（珠海学院教授）

## 许可证

MIT
