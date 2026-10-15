# Growth Experiment Lab

Growth Experiment Lab 是一个个人开源求职作品集项目，用于演示用户增长场景中的
A/B 实验设计、质量检查、效果评估与基于 IPTW 的观察性因果分析。

本项目第一版只使用带固定随机种子的**合成数据**。它不是任何公司的真实生产系统，
也不包含真实用户、真实业务或线上实验结果。仓库中的任何指标、图表和结论都必须由
代码运行产生，不会预填或伪造分析结果。

## 当前状态

目前处于阶段 1：工程骨架与统计契约。当前版本只包含配置读取、配置校验、基础 CLI
和文档框架，尚未实现数据生成、A/B 分析、CUPED 或 IPTW。

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
```

未来阶段计划提供一条端到端命令，但在实现前不会展示虚构输出。

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

