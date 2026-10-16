# CUPED

CUPED 第一版只用于 `watch_time_minutes`，唯一协变量为处理前
`pre_7d_watch_time_minutes`。在全部随机用户上统一估计：

```text
theta = cov(Y, X) / var(X)
Y_cuped = Y - theta * (X - mean(X))
```

不能按处理组分别拟合 theta，也不能使用实验期间或其他处理后变量。CUPED 的作用是利用
处理前预测信息降低方差，不是制造更大的处理效果。协变量相关性较弱时，方差降低可能
很小；零方差协变量会明确失败。
