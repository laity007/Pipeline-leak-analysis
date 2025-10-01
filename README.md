# Pipeline-leak-analysis

管道泄漏贝叶斯分析（MMSE 卡尔曼平滑 vs MAP 稀疏+平滑）。

## 环境准备

```bash
python -m venv .venv  # 可选
source .venv/bin/activate
pip install numpy matplotlib
```

## 复现实验

1. 运行指标脚本（默认数据：`data2/(1).csv`）：
   ```bash
   python scripts/run_mmse_vs_map.py
   ```
   * 指标保存至 `plot/(1)/metrics_channel_k.json`。

2. 在本地生成图像：
   * 进入 `plot/(1)/`，运行任意脚本，例如：
     ```bash
     python plot/(1)/plot_channel_1_A1_observed_vs_kalman.py
     python plot/(1)/plot_channel_1_C2_metrics.py
     ```
   * 每个脚本会自动读取数据、重新计算模型并通过 `matplotlib` 展示图像，用户可在交互界面保存 PNG。

## 主要脚本

| 路径 | 功能 |
| --- | --- |
| `analysis/mmse_map.py` | 数据加载、卡尔曼 EM 平滑、MAP 交替最小化、指标计算。 |
| `analysis/plotting.py` | 通用绘图函数，供 `plot/(1)/plot_channel_*` 脚本调用。 |
| `scripts/run_mmse_vs_map.py` | 运行全部通道的 MMSE vs MAP 分析并输出 JSON 指标。 |
| `REPORT.md` | 深度诊断与实验结论。 |

## 注意事项

* `data2/(1).csv` 的时间戳存在大量重复（详见 `REPORT.md`），模型会自动给出告警指标。
* 若需分析其他 CSV，可在命令或脚本中修改路径。
* 旧版 MATLAB 绘图脚本仍保留，但推荐使用新的 Python 流程以保证可复现性。
