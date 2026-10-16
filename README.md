# Growth Experiment Lab

Growth Experiment Lab 是一个个人开源求职作品集项目，用于演示用户增长场景中的
A/B 实验设计、质量检查、效果评估与基于 IPTW 的观察性因果分析。

本项目第一版只使用带固定随机种子的**合成数据**。它不是任何公司的真实生产系统，
也不包含真实用户、真实业务或线上实验结果。仓库中的任何指标、图表和结论都必须由
代码运行产生，不会预填或伪造分析结果。

## 当前状态

目前完成至阶段 4：工程骨架、合成数据、DuckDB 用户级底表、数据质量检查、实验设计、
随机化诊断和随机 A/B 效果评估。观察性 propensity score 与 IPTW 尚未实现。

## 环境

- Python 3.11
- Conda 环境名称：`growth_exp`（项目不会创建、删除或切换环境）
- 运行命令优先采用 `python -m ...`

在已激活的 `growth_exp` 环境中，安装项目及开发依赖的建议命令为：

```bash
python -m pip install -e ".[dev]"
```

请仅在确认需要安装依赖后手动运行上述命令。本项目不会自动安装、升级或卸载包。

## 当前可用命令

```bash
python -m growth_experiment_lab --help
python -m growth_experiment_lab validate-config --config configs/default.toml
python -m growth_experiment_lab generate-data --config configs/default.toml
python -m growth_experiment_lab build-mart --config configs/default.toml
python -m growth_experiment_lab validate-data --config configs/default.toml
python -m growth_experiment_lab plan-experiment --config configs/default.toml
python -m growth_experiment_lab diagnose-randomization --config configs/default.toml
python -m growth_experiment_lab analyze-ab --config configs/default.toml
python -m growth_experiment_lab validate-simulation-estimates --config configs/default.toml
```

数据文件默认写入 `data/raw/`，真实处理效应元数据写入
`data/metadata/true_effects.json`。如文件已存在，命令会拒绝覆盖；只有显式添加
`--overwrite` 才会替换。完整生成数据默认不纳入版本控制。

`build-mart` 同时构建随机实验和观察性用户级底表，默认拒绝覆盖，可显式添加
`--overwrite`。`validate-data` 验证原始表与底表，并将机器可读摘要写入
`reports/generated/data_quality_summary.json`。

生成器同时创建随机实验数据和带可观测混杂的观察性数据。其结构与生成方法见
[`docs/synthetic_data.md`](docs/synthetic_data.md)，字段定义见
[`docs/data_dictionary.md`](docs/data_dictionary.md)。
数据质量与泄漏边界见 [`docs/data_quality.md`](docs/data_quality.md)。
实验设计与诊断方法分别见 [`docs/experiment_design.md`](docs/experiment_design.md) 和
[`docs/randomization_diagnostics.md`](docs/randomization_diagnostics.md)。
正式 ITT 估计方法见 [`docs/ab_effect_estimation.md`](docs/ab_effect_estimation.md)。

`analyze-ab` 只读取 randomized mart，不读取合成真值。独立的
`validate-simulation-estimates` 仅在正式结果生成后读取真值，用于验证合成数据估计误差，
其输出不得进入业务决策支持。

## 验证

```bash
python -m pytest
python -m ruff check .
```

## 核心分析约定

- 实验单位：`user_id`
- 分析原则：按首次随机分组进行 ITT 分析
- 主指标：`next_day_retention`
- 次要指标族：`clicked`、`watch_time_minutes`，采用 Holm 校正
- 护栏指标：`negative_feedback`，最大允许恶化为 0.005
- 唯一预定义分群：`is_new_user`
- 观察性分析目标：ATE；主要方法为 stabilized IPTW

完整约定见 [PROJECT_SPEC.md](PROJECT_SPEC.md)，方法假设与局限见 `docs/`。

## 许可证

MIT License，详见 [LICENSE](LICENSE)。

## 观察性因果分析

```bash
python -m growth_experiment_lab analyze-observational --config configs/default.toml
python -m growth_experiment_lab validate-observational-simulation --config configs/default.toml
```

该流程仅分析合成的观察性数据，报告 naive difference、stabilized IPTW ATE、overlap、
权重稳定性、ESS、加权前后 SMD、预先声明的 1%/99% 权重截断敏感性分析，以及每次
重新拟合倾向模型的用户级 Bootstrap 区间。结果不代表真实业务效果，且必须人工复核。
