# Growth Experiment Lab

Growth Experiment Lab 是一个个人开源求职作品集项目，用于演示用户增长场景中的
A/B 实验设计、质量检查、效果评估与基于 IPTW 的观察性因果分析。

本项目第一版只使用带固定随机种子的**合成数据**。它不是任何公司的真实生产系统，
也不包含真实用户、真实业务或线上实验结果。仓库中的任何指标、图表和结论都必须由
代码运行产生，不会预填或伪造分析结果。

## 当前状态

目前完成至阶段 2A：工程骨架、统计契约和可复现的合成数据生成。当前版本尚未实现
DuckDB 分析底表、A/B 显著性检验、CUPED 或 IPTW 估计。

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
```

数据文件默认写入 `data/raw/`，真实处理效应元数据写入
`data/metadata/true_effects.json`。如文件已存在，命令会拒绝覆盖；只有显式添加
`--overwrite` 才会替换。完整生成数据默认不纳入版本控制。

生成器同时创建随机实验数据和带可观测混杂的观察性数据。其结构与生成方法见
[`docs/synthetic_data.md`](docs/synthetic_data.md)，字段定义见
[`docs/data_dictionary.md`](docs/data_dictionary.md)。

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
