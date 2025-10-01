% plot_signals_with_smooth.m
% 绘制三个通道的波形 + 平滑滤波曲线，辅助识别泄漏信号

clear; clc;

% === 1) 读取预处理后的数据 ===
inputCsv = 'data2/(15).csv';   % 修改为你的文件路径
T = readtable(inputCsv);

time_sec = T.time_sec;
y1 = T.channel_1;
y2 = T.channel_2;
y3 = T.channel_3;

% === 2) 设置平滑参数 ===
window_size = 20; % 移动平均窗口大小，可根据采样率调节
smooth1 = movmean(y1, window_size);
smooth2 = movmean(y2, window_size);
smooth3 = movmean(y3, window_size);

% === 3) 绘制三通道波形 ===
figure('Color','w','Position',[100 100 800 600]);

subplot(3,1,1);
plot(time_sec, y1, 'b-', 'DisplayName','原始信号'); hold on;
plot(time_sec, smooth1, 'r-', 'LineWidth',1.2, 'DisplayName','平滑趋势');
grid on; xlabel('时间 / 秒'); ylabel('幅值 / 电压');
title('通道 1 波形'); legend('Location','best');

subplot(3,1,2);
plot(time_sec, y2, 'b-', 'DisplayName','原始信号'); hold on;
plot(time_sec, smooth2, 'r-', 'LineWidth',1.2, 'DisplayName','平滑趋势');
grid on; xlabel('时间 / 秒'); ylabel('幅值 / 电压');
title('通道 2 波形'); legend('Location','best');

subplot(3,1,3);
plot(time_sec, y3, 'b-', 'DisplayName','原始信号'); hold on;
plot(time_sec, smooth3, 'r-', 'LineWidth',1.2, 'DisplayName','平滑趋势');
grid on; xlabel('时间 / 秒'); ylabel('幅值 / 电压');
title('通道 3 波形'); legend('Location','best');

% === 4) 保存图像 ===
[~,fname,~] = fileparts(inputCsv);
outDir = fullfile('plot','plot_waveform_smooth');
if ~exist(outDir,'dir'), mkdir(outDir); end
outPath = fullfile(outDir, [fname '_waveform_smooth.png']);
exportgraphics(gcf, outPath, 'Resolution',150);

fprintf('波形图(含平滑曲线)已保存到: %s\n', outPath);
