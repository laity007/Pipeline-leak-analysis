function plot_all_figures(t, y, x_kal, residA, x_map, r_map, residB, outDir, chIdx)
% 为单个通道生成 6 张图（中文）并保存
% A1: 观测 vs 卡尔曼
% A2: 卡尔曼残差
% B1: 观测 vs MAP（平滑+稀疏）
% B2: MAP 残差
% C1: Kalman vs MAP 最终估计
% C2: 指标对比（SNR 提升、残差方差）

% 计算指标
snrA = 10*log10(var(x_kal) / var(residA));
snrB = 10*log10(var(x_map + r_map) / var(residB));
rvA  = var(residA);
rvB  = var(residB);

% 文件名前缀
pf = @(name) fullfile(outDir, sprintf('channel_%d_%s.png', chIdx, name));

% A1
figure('Color','w'); hold on;
plot(t, y, 'DisplayName','观测信号');
plot(t, x_kal, 'LineWidth',1.2, 'DisplayName','卡尔曼平滑 (MMSE)');
xlabel('时间 / 秒'); ylabel('幅值 / 电压');
title(sprintf('通道 %d：观测 vs 卡尔曼平滑', chIdx));
legend('Location','best'); grid on;
exportgraphics(gcf, pf('A1_observed_vs_kalman'), 'Resolution',150); close;

% A2
figure('Color','w');
plot(t, residA, 'DisplayName','残差'); 
xlabel('时间 / 秒'); ylabel('幅值 / 电压');
title(sprintf('通道 %d：卡尔曼残差（应近似白噪声）', chIdx));
grid on;
exportgraphics(gcf, pf('A2_kalman_residual'), 'Resolution',150); close;

% B1
figure('Color','w'); hold on;
plot(t, y, 'DisplayName','观测信号');
plot(t, x_map, 'LineWidth',1.2, 'DisplayName','MAP 平滑成分 x');
plot(t, r_map, 'LineWidth',1.2, 'DisplayName','MAP 稀疏成分 r');
xlabel('时间 / 秒'); ylabel('幅值 / 电压');
title(sprintf('通道 %d：MAP 分解（平滑 + 稀疏）', chIdx));
legend('Location','best'); grid on;
exportgraphics(gcf, pf('B1_observed_vs_map_components'), 'Resolution',150); close;

% B2
figure('Color','w');
plot(t, residB, 'DisplayName','残差');
xlabel('时间 / 秒'); ylabel('幅值 / 电压');
title(sprintf('通道 %d：MAP 残差（应近似白噪声）', chIdx));
grid on;
exportgraphics(gcf, pf('B2_map_residual'), 'Resolution',150); close;

% C1
figure('Color','w'); hold on;
plot(t, x_kal, 'LineWidth',1.2, 'DisplayName','卡尔曼 MMSE');
plot(t, x_map + r_map, 'LineWidth',1.2, 'DisplayName','MAP (x+r)');
xlabel('时间 / 秒'); ylabel('幅值 / 电压');
title(sprintf('通道 %d：两种估计对比（卡尔曼 vs MAP）', chIdx));
legend('Location','best'); grid on;
exportgraphics(gcf, pf('C1_kalman_vs_map'), 'Resolution',150); close;

% C2 指标对比（条形图）
figure('Color','w');
subplot(2,1,1);
bar([snrA, snrB]); 
set(gca,'XTickLabel',{'卡尔曼','MAP'});
ylabel('SNR 提升 / dB'); title(sprintf('通道 %d：SNR 提升对比', chIdx));
grid on;

subplot(2,1,2);
bar([rvA, rvB]);
set(gca,'XTickLabel',{'卡尔曼','MAP'});
ylabel('残差方差'); title(sprintf('通道 %d：残差方差对比', chIdx));
grid on;

exportgraphics(gcf, pf('C2_metrics_compare'), 'Resolution',150); close;
end
