function [x, r, resid, exitMsg] = map_sparse_smooth_qp(y, lambda_smooth, lambda_sparse)
% 稀疏+平滑 MAP：
%   min_{x,r}  1/2||y - x - r||^2 + (lambda_smooth/2)||D x||^2 + lambda_sparse*||r||_1
% 用 QP 形式通过 quadprog 求解：
%   变量 z = [x; r; u], u >= |r| 作为 L1 的辅助变量
%   目标: 0.5 z^T H z + f^T z
%   约束: r - u <= 0,  -r - u <= 0
%
% 优点（相对手写迭代/ADMM）：
%   - 由优化器保证凸性与数值稳健性
%   - 收敛判据、停止条件、约束处理更成熟
%   - 便于扩展其他约束（如边界、平滑度上下限等）

N = numel(y);

% 构造差分矩阵 D（(N-1)xN）
e = ones(N,1);
D = spdiags([-e e],[0 1], N-1, N);

% 构造 H, f
% 目标: 0.5*(x^T (I+λ D^T D) x + r^T I r - 2 x^T r ) - y^T x - y^T r + λ1 * 1^T u + const
Axx = speye(N) + lambda_smooth*(D'*D);
Arr = speye(N);
Axr = -speye(N); % x^T*(-I)*r  → off-diagonal

H = [Axx,  Axr,  sparse(N,N);
     Axr', Arr,  sparse(N,N);
     sparse(N,N), sparse(N,N), sparse(N,N)];  % u 部分无二次项

f = [-y; -y; lambda_sparse*ones(N,1)];

% 线性不等式约束： r - u <= 0  和  -r - u <= 0
% 变量顺序 [x; r; u]
Aineq = [sparse(N,N),  speye(N), -speye(N);
         sparse(N,N), -speye(N), -speye(N)];
bineq = zeros(2*N,1);

% 无等式约束
Aeq = []; beq = [];

% 无显式边界（u>=0 已由两组不等式隐含）
lb = []; ub = [];

% 调用 quadprog
opts = optimoptions('quadprog','Algorithm','interior-point-convex', ...
    'Display','off','MaxIterations',200, 'OptimalityTolerance',1e-8);

[z, ~, exitflag, output] = quadprog(H, f, Aineq, bineq, Aeq, beq, lb, ub, [], opts);

exitMsg = '';
if exitflag <= 0
    exitMsg = output.message;
end

% 拆回 x, r, u
x = z(1:N);
r = z(N+1:2*N);
% u = z(2*N+1:end); %#ok<NASGU>

resid = y - x - r;
end
