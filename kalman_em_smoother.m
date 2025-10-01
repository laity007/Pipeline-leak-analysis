function [x_smooth, P_smooth, Q_hat, R_hat, resid] = kalman_em_smoother(y, dt, opts)
% Local-Level 模型 + RTS 平滑 + EM 学习 Q,R
% 状态: x_t = x_{t-1} + w_t,  w ~ N(0, Q*dt)  (dt 自适应缩放)
% 观测: y_t = x_t + v_t,      v ~ N(0, R)
%
% 输入:
%   y   : Nx1 观测
%   dt  : 采样间隔（标量）
%   opts.maxEmIters (默认 15)
%   opts.verbose    (默认 false)
%
% 输出:
%   x_smooth : Nx1 平滑后验均值
%   P_smooth : Nx1 平滑后验方差
%   Q_hat, R_hat : EM 学到的噪声方差
%   resid : Nx1 残差 y - x_smooth

N = numel(y);
if ~isfield(opts, 'maxEmIters'), opts.maxEmIters = 15; end
if ~isfield(opts, 'verbose'),     opts.verbose = false; end

% 初值
x0 = y(1);
P0 = var(y);
R_hat = 0.1*var(y);     %%%%%%%%%%%%%%%%%%%%%%%%  减小，允许状态更快变化，卡尔曼能跟随信号
Q_hat = 0.05*var(y);    %%%%%%%%%%%%%%%%%%%%%%%%  增大，增加观测数据权重，减少贴近零线问题

A = 1; H = 1;

for it = 1:opts.maxEmIters
    % Forward KF
    x_pred = zeros(N,1); P_pred = zeros(N,1);
    x_filt = zeros(N,1); P_filt = zeros(N,1);

    x_prev = x0; P_prev = P0;
    for t = 1:N
        % 预测
        x_pred(t) = A*x_prev;
        P_pred(t) = A*P_prev*A' + Q_hat*dt;

        % 更新
        S = H*P_pred(t)*H' + R_hat;
        K = P_pred(t)*H'/S;
        x_filt(t) = x_pred(t) + K*(y(t) - H*x_pred(t));
        P_filt(t) = (1 - K*H)*P_pred(t);

        x_prev = x_filt(t);
        P_prev = P_filt(t);
    end

    % RTS 平滑
    x_smooth = zeros(N,1); P_smooth = zeros(N,1);
    C = zeros(N-1,1); % lag-one covariance

    x_smooth(N) = x_filt(N); P_smooth(N) = P_filt(N);
    for t = N-1:-1:1
        J = P_filt(t)*A' / P_pred(t+1);
        x_smooth(t) = x_filt(t) + J*(x_smooth(t+1) - x_pred(t+1));
        P_smooth(t) = P_filt(t) + J*(P_smooth(t+1) - P_pred(t+1))*J';
        C(t) = J * P_smooth(t+1);
    end

    % EM 更新 Q,R
    Exx    = P_smooth + x_smooth.^2;
    sum_Exx     = sum(Exx);
    sum_Exx_lag = sum(P_smooth(1:end-1) + x_smooth(1:end-1).^2);
    sum_cross   = sum(C + x_smooth(1:end-1).*x_smooth(2:end));

    Q_hat = (sum(Exx(2:end)) - 2*sum_cross + sum_Exx_lag) / (N-1);
    Q_hat = max(Q_hat, 1e-12);

    R_hat = sum((y - x_smooth).^2 + P_smooth) / N;
    R_hat = max(R_hat, 1e-12);

    x0 = x_smooth(1); P0 = P_smooth(1);

    if opts.verbose
        fprintf('  EM iter %2d: Q=%.6g, R=%.6g\n', it, Q_hat, R_hat);
    end
end

% 最终再跑一次 KF+RTS（可复用）
x_pred = zeros(N,1); P_pred = zeros(N,1);
x_filt = zeros(N,1); P_filt = zeros(N,1);
x_prev = x0; P_prev = P0;
for t = 1:N
    x_pred(t) = A*x_prev;
    P_pred(t) = A*P_prev*A' + Q_hat*dt;
    S = H*P_pred(t)*H' + R_hat;
    K = P_pred(t)*H'/S;
    x_filt(t) = x_pred(t) + K*(y(t) - H*x_pred(t));
    P_filt(t) = (1 - K*H)*P_pred(t);
    x_prev = x_filt(t); P_prev = P_filt(t);
end
x_smooth = zeros(N,1); P_smooth = zeros(N,1);
x_smooth(N) = x_filt(N); P_smooth(N) = P_filt(N);
for t = N-1:-1:1
    J = P_filt(t)/P_pred(t+1);
    x_smooth(t) = x_filt(t) + J*(x_smooth(t+1) - x_pred(t+1));
    P_smooth(t) = P_filt(t) + J*(P_smooth(t+1) - P_pred(t+1))*J';
end

resid = y - x_smooth;
end
