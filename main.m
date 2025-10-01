% main.m
% 仅处理 data2 文件夹下的一个指定文件，生成每个通道的 6 张中文图
% 依赖文件：load_standard_csv.m, kalman_em_smoother.m, map_sparse_smooth_qp.m,
%           make_chinese.m, plot_all_figures.m
clear all; clc;

% === 1) 指定输入文件（标准化格式：time_sec, channel_1, channel_2, channel_3）===
inputCsv = 'data2/(1).csv';  % ← 修改为你的文件
assert(exist(inputCsv, 'file')==2, '找不到输入文件：%s', inputCsv);

% === 2) 载入数据 ===
[time_sec, data, filePrefix] = load_standard_csv(inputCsv); % data: N x 3
fprintf('读取完成：%s，样本数 N=%d，通道数=%d\n', inputCsv, size(data,1), size(data,2));

% 输出路径
outDir = fullfile('plot', filePrefix);
if ~exist(outDir, 'dir'); mkdir(outDir); end

% 检查采样间隔（你已确认等间隔）
dt = mean(diff(time_sec));
if any(abs(diff(time_sec)-dt) > 1e-9)
    warning('检测到非等间隔时间戳，将采用 dt 自适应。');
end
fprintf('估计采样间隔 dt = %.6f 秒\n', dt);

% === 3) 超参数（自动设定；可在此集中调整系数）===
c_sparse = 2.5;   % λ_sparse = c_sparse * sigma_MAD
c_smooth = 10.0;  % λ_smooth = c_smooth / dt

% === 4) 循环处理每个通道 ===
for k = 1:size(data,2)
    y = data(:,k);

    % —— A) 卡尔曼 + RTS + EM 学习 Q,R ——
    fprintf('\n[通道 %d] 卡尔曼EM估计开始...\n', k);
    optsA.maxEmIters = 10;      %%%%%%%%%%%%%%%%%%%%% 减小，防止 Q 收敛过小，减少过度平滑
    optsA.verbose    = true;  % 控制台打印 Q,R
    [x_kal, P_kal, Q_hat, R_hat, residA] = kalman_em_smoother(y, dt, optsA);
    fprintf('[通道 %d] EM 结束：Q=%.6g, R=%.6g\n', k, Q_hat, R_hat);

    % —— B) 稀疏+平滑 MAP（QP）——
    % 自动设定超参数（鲁棒噪声估计）
    sigma_mad = median(abs(y - median(y))) / 0.6745;
    lambda_sparse = c_sparse * sigma_mad;
    lambda_smooth = c_smooth / max(dt, eps);
    fprintf('[通道 %d] MAP/QP 超参数：lambda_smooth=%.6g, lambda_sparse=%.6g\n', ...
            k, lambda_smooth, lambda_sparse);

    [x_map, r_map, residB, qpInfo] = map_sparse_smooth_qp(y, lambda_smooth, lambda_sparse);
    if ~isempty(qpInfo)
        fprintf('[通道 %d] quadprog 退出信息：%s\n', k, qpInfo);
    end

    % —— C) 生成并保存 6 张中文图 —— 
    make_chinese(); % 设置中文字体（Microsoft YaHei 等回退）
    plot_all_figures(time_sec, y, x_kal, residA, x_map, r_map, residB, ...
                     outDir, k);
end

fprintf('\n全部完成。输出目录：%s\n', outDir);
