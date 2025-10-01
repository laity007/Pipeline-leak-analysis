% plot_signals.m
% 绘制三个通道的波形，用于初步判断是否存在泄漏信号

clear; clc;

% === 1) 读入预处理后的数据文件 ===
inputCsv = 'data2/(4).csv';   % 修改为你的文件路径
T = readtable(inputCsv);

time_sec = T.time_sec;
y1 = T.channel_1;
y2 = T.channel_2;
y3 = T.channel_3;

% === 2) 绘制三个通道波形 ===
figure('Color','w','Position',[100 100 800 600]);

subplot(3,1,1);
plot(time_sec, y1, 'b-'); grid on;
xlabel('时间 / 秒'); ylabel('幅值 / 电压');
title('通道 1 波形');

subplot(3,1,2);
plot(time_sec, y2, 'r-'); grid on;
xlabel('时间 / 秒'); ylabel('幅值 / 电压');
title('通道 2 波形');

subplot(3,1,3);
plot(time_sec, y3, 'g-'); grid on;
xlabel('时间 / 秒'); ylabel('幅值 / 电压');
title('通道 3 波形');

% === 3) 保存图像 ===
[~,fname,~] = fileparts(inputCsv);
outDir = fullfile('plot','plot_waveform');
if ~exist(outDir,'dir'), mkdir(outDir); end
outPath = fullfile(outDir, [fname '_waveform.png']);
exportgraphics(gcf, outPath, 'Resolution',150);

fprintf('波形图已保存到: %s\n', outPath);
