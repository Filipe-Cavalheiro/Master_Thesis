clear; clc; close all;
% Parameters
param.nD = 4;              % drones
param.Tend = 10;
param.dt = 0.1;
param.g = 9.81;
param.Nsim = param.Tend/param.dt;     % alternating-minimization iterations
param.traj_points = 1000;


% Collision avoidance params
bf.w_g = 7000;         % Goal gain
bf.w_s = 100;          % smoothness cost
bf.K = 30;             % Prediction horizon
bf.k = 5;              % Penalty horizon
bf.n = 10;             % Bernstein degree

max_vel = 1.73;
min_acc = 0.3*param.g;
max_acc = 1.5*param.g;
theta_ij = diag([0.17, 0.17, 0.45]);
theta_col = diag([0.13, 0.13, 0.40]);
theta_p = diag([0.2, 0.2, 0.2]);

% Start / Goal positions
p_0 = [-4 -4 1;
       -4  4 1;
        4 -4 1;
        4  4 1]';

p_d = [ 4  4 1;
        4 -4 1;
       -4  4 1;
       -4 -4 1]';

% Preallocate cell array
chaser = cell(1, param.nD);

% Template struct
template = struct( ...
    'pos',      zeros(3, param.Nsim+1), ...
    'vel',      zeros(3, param.Nsim+1), ...
    'acc',  zeros(3, param.Nsim+1), ...
    'attitude', zeros(9, param.Nsim+1), ...
    'desired_pos', zeros(3,1) ...
    );

% Allocate all drones at once
[chaser{:}] = deal(template);
for iD = 1:param.nD
    chaser{iD}.pos(:,1) = p_0(:,iD);
    chaser{iD}.vel(:,1) = [0;0;0];
    chaser{iD}.acc(:,1) = [0;0;0];
    chaser{iD}.alpha(:,1) = 0;
    chaser{iD}.desired_pos = p_d(:,iD);
end

% Initial trajectories (straight lines)
traj = zeros(param.traj_points,3,param.nD);
for i=1:param.nD
    traj(:,1,i) = linspace(p_0(1,i),p_d(1,i),param.traj_points);
    traj(:,2,i) = linspace(p_0(2,i),p_d(2,i),param.traj_points);
    traj(:,3,i) = linspace(p_0(3,i),p_d(3,i),param.traj_points);
end

residual = inf;
threshold = 1e-2;
iter = 1;
% Alternating minimization
while iter < param.Nsim || residual >= threshold 
    fprintf('AM iteration %d\n',iter);
    for iD = 1:param.nD
            % Bernstein basis matrix            
            syms bernstein_basis
            
            for m = 0:bf.n
                B(m+1) = nchoosek(bf.n,m)*(1-bernstein_basis)^(bf.n-m)*bernstein_basis^m;
            end
            
            dB  = diff(B,bernstein_basis);
            ddB = diff(B,bernstein_basis,2);
            
            sgrid = linspace(0,1,bf.K);
            
            W   = double(subs(B,bernstein_basis,sgrid)).';
            Wd  = double(subs(dB,bernstein_basis,sgrid)).';
            Wd  = Wd  / param.Tend;
            Wdd = double(subs(ddB,bernstein_basis,sgrid)).';
            Wdd = Wdd / param.Tend^2;

            F_c = kron(eye(2*param.nD),W);
            A_tilde = [Wd, Wdd, F_c];
            A = blkdiag(A_tilde, A_tilde, A_tilde);
            % S1 
            Abar = Q + rho*(A'*A);
            
            bbar = -q ...
                   + rho*A'*b ...
                   - lambda;
            
            KKT = [
                Abar C';
                C zeros(size(C,1))
            ];
            
            rhs = [
                bbar;
                e
            ];
            
            sol = KKT\rhs;
            
            zeta_1 = sol(1:size(Q,1));
        
            % S2
            [alpha,beta] = solveS2(zeta1,d);
        
            % S3
            c1 = a^2*cos(alpha).^2 + b^2*sin(alpha).^2;
            
            c2 = a*(x-xobs).*cos(alpha) + b*(y-yobs).*sin(alpha);
            
            d = c2./c1;
            
            d = max(d,1);
        
            % S4
            residual = -G*zeta_1 + h;
            
            s = max(0,residual);

            % S5 
            lambda = lambda - rho*A'*residual;
    end
    iter = iter + 1;
end