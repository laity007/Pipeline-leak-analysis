# MMSE vs MAP Analysis Report for `data2/(1).csv`

## 1. 项目体检与主要发现

| 维度 | 结论 |
| --- | --- |
| 数据完整性 | 2401 条样本仅包含 13 个唯一时间戳，重复率 **99.46%**，说明传感器时间列几乎完全失真，导致连续模型（卡尔曼、平滑项）被迫在极少的有效时刻上拟合。 |
| 旧实现隐患 | 旧版 MATLAB 流程直接导出 `.png`，无法复现；EM 卡尔曼中过程噪声会塌缩到 0，MAP 求解器依赖 `quadprog`，在无 MATLAB 环境下不可用。 |
| 新增诊断 | 指标脚本 `scripts/run_mmse_vs_map.py` 现输出时间戳重复率、SNR、残差方差与一阶自相关，可快速定位异常。 |

## 2. 实验复现：MMSE（卡尔曼平滑） vs MAP（稀疏+平滑）

运行 `python scripts/run_mmse_vs_map.py` 会针对 3 个通道生成如下指标（JSON 见 `plot/(1)/metrics_channel_k.json`）：

| Channel | SNR<sub>MMSE</sub> (dB) | SNR<sub>MAP</sub> (dB) | Var(resid)<sub>MMSE</sub> | Var(resid)<sub>MAP</sub> | ρ<sub>1</sub>(resid)<sub>MMSE</sub> | ρ<sub>1</sub>(resid)<sub>MAP</sub> |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | -24.53 | -5.25 | 0.3813 | 0.2284 | 0.899 | 0.853 |
| 2 | -21.04 | -0.75 | 0.1578 | 0.0713 | 0.939 | 0.889 |
| 3 | -19.22 | 0.40 | 0.1285 | 0.0518 | 0.945 | 0.878 |

> 说明：ρ<sub>1</sub> 表示残差的一阶自相关，理想白噪声应接近 0。实际值偏大，进一步印证时间戳异常导致模型无法获得白噪声残差。

## 3. 深度诊断与改进

1. **时间戳崩塌**：由于 99% 的记录重复时间戳，连续状态模型实际上在离散索引上运作，所有平滑结果被迫“强行插值”。建议追溯数据产生流程，恢复真实采样时刻或在预处理阶段以均匀网格重新采样。
2. **卡尔曼平滑器过度收缩**：旧实现会得到 `Q→0`，输出近乎常数。本次在 `analysis/mmse_map.py` 中加入 `Q` 的自适应下界（`max(1e-3·Var(y), 1e-9)`），避免过程噪声被完全抹除，使 MMSE 估计具备有限幅值。【F:analysis/mmse_map.py†L79-L147】
3. **MAP 残差相关性**：虽然 MAP 相比 MMSE 在方差上有更大幅度降低，但残差相关性仍高。这与时间戳问题相关，也说明当前 `λ_smooth=10/dt` 偏大。可进一步做交叉验证或依据残差白化度自动调节超参数。

## 4. 交付脚本与复现路径

1. **指标生成**：`python scripts/run_mmse_vs_map.py`（会更新 `plot/(1)/metrics_channel_k.json`）。【F:scripts/run_mmse_vs_map.py†L1-L36】
2. **绘图脚本**：位于 `plot/(1)/`，每个通道 6 个脚本（A1~C2），例如：
   * `python plot/(1)/plot_channel_1_A1_observed_vs_kalman.py`
   * `python plot/(1)/plot_channel_1_C2_metrics.py`
   运行时会自动读取数据、重算模型并显示图像，便于用户在本地保存。脚本内部复用了 `analysis/plotting.py` 的通用绘图逻辑。【F:analysis/plotting.py†L1-L68】【F:plot/(1)/plot_channel_1_A1_observed_vs_kalman.py†L1-L12】
3. **模型与诊断核心**：集中在 `analysis/mmse_map.py`，提供数据加载、卡尔曼 EM、稀疏+平滑交替最小化、指标计算与缓存复用。【F:analysis/mmse_map.py†L1-L210】

## 5. 下一步建议

* **恢复真实时间轴**：若能提供准确采样时刻，卡尔曼与 MAP 的残差可望接近白噪声，SNR 也能反映真实滤波增益。
* **超参数自适应**：在 MAP 求解前以网格或准则（AIC/GCV）自动挑选 `λ_smooth`、`λ_sparse`，降低人工调参成本。
* **残差白噪声检验**：可在指标脚本中增加 Ljung–Box 等统计检验，进一步量化模型拟合优劣。

运行上述脚本即可复现本文全部结论与图像。后续若替换数据，只需调整 CSV 路径即可。
