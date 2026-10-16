# 合成数据字典

## users

该表只允许包含处理前变量。

| 字段 | 类型 | 含义 |
|---|---|---|
| `user_id` | string | 数据集内唯一的合成用户标识 |
| `assignment_date` | date/time | 合成入组日期 |
| `device_type` | category | 入组前设备类型 |
| `region` | category | 入组前地区类别 |
| `is_new_user` | integer 0/1 | 实验开始前定义的新用户标记 |
| `pre_7d_sessions` | non-negative integer | 入组前 7 日会话数 |
| `pre_7d_watch_time_minutes` | non-negative float | 入组前 7 日观看分钟数 |
| `pre_7d_click_days` | integer 0–7 | 入组前 7 日发生点击的天数 |
| `pre_experiment_activity_score` | float | 仅由处理前历史活动构造的标准化分数 |

## assignment

| 字段 | 类型 | 含义 |
|---|---|---|
| `user_id` | string | 合成用户标识 |
| `treatment` | integer 0/1 | 实际处理分配 |
| `assignment_probability` | float (0, 1) | 生成处理时使用的真实概率 |
| `dataset_type` | string | `randomized` 或 `observational` |

## outcomes

| 字段 | 类型 | 含义 |
|---|---|---|
| `user_id` | string | 合成用户标识 |
| `clicked` | integer 0/1 | 实验期内是否至少点击一次 |
| `watch_time_minutes` | non-negative float | 实验期观看分钟数，保留零值用户 |
| `next_day_retention` | integer 0/1 | 入组次日是否留存 |
| `negative_feedback` | integer 0/1 | 实验期内是否产生负反馈 |

`assignment_probability` 是合成数据生成机制的一部分，不代表真实研究中可以观察到真实
倾向得分。正式倾向模型只能使用 `users` 中预先声明的处理前变量，不能使用 outcomes。

## experiment_mart

正式用户级分析底表一名用户一行，字段白名单为：

- 标识与分配：`user_id`、`assignment_date`、`dataset_type`、`treatment`。
- 处理前协变量：`device_type`、`region`、`is_new_user`、`pre_7d_sessions`、
  `pre_7d_watch_time_minutes`、`pre_7d_click_days`、`pre_experiment_activity_score`。
- 观测结果：`clicked`、`watch_time_minutes`、`next_day_retention`、
  `negative_feedback`。

正式处理前协变量白名单仅为上述七个处理前字段。结果字段不能作为处理前协变量。
`assignment_probability`、`p0`、`p1`、真实效果、潜在结果、内部潜变量以及
`true_effects.json` 中的值明确禁止进入底表。
