# Changelog

## [Unreleased]
- 替换旧的 MATLAB 批量绘图流程，新增 Python 分析模块 `analysis/mmse_map.py` 与绘图助手 `analysis/plotting.py`。
- 新增 `scripts/run_mmse_vs_map.py` 生成每个通道的 MMSE vs MAP 指标 JSON。
- 将 `plot/(1)/` 下的 PNG 改为可复现的绘图脚本 `plot_channel_*_*.py`。
- 输出实验诊断报告 `REPORT.md`，更新 `README.md` 与 `.gitignore`，说明复现步骤并忽略临时图像。
