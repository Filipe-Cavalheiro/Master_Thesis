%The programs purpose is to in a spherical surface using a
%quadratic program clf and cbf having the objective of not colliding with
%obstacles while doing so. 
%Multiple simulations are ran where the obstacle position is varied

clc; clear all; close all;

% === Simulation parameters ===
Tend = 10;
dt   = 0.005;
Nsim = round(Tend/dt) + 1;
t    = linspace(0, Tend, Nsim);

% === Parameters CBF ===
% Parameters for the circle

y = 180;  % Latitude in degrees
r_safe = 0.2;
theta_safe = r_safe*pi;
y_rad = deg2rad(y);
center_z = sin(y_rad);

k0_cbf = 2;     % CBF gain 1st order
k1_cbf = 2;     % CBF gain 2nd order
k_clf = 0.015;     % CLF gain 1st order

% System parameters (V. Numerical Simulations)
c = 0.3;                             % Nm (s/rad)
J = diag([0.0294 0.0305 0.0495]);    % kg m^2
tau = zeros(3,1);

% Controller gains (29a, 29b)
K_r = diag([3 3 3]);
K_w = diag([7.056 7.320 11.88]) * 1e-1;

% Hat (skew) operator
S = @(x) [   0   -x(3)  x(2);
           x(3)    0   -x(1);
          -x(2)  x(1)    0 ];

% Initial conditions
Q = eye(3);
omega_d = zeros(3,1);

steps = length(0:10:360); % clearer
steps = 2;
fixed_obstacle = [0; -1; 0];
fixed_obstacle = fixed_obstacle / norm(fixed_obstacle);
moving_obstacle = [-1;0;0]; %just initialization

idx.moving = 1;
idx.fixed  = 2;
obstacles = {fixed_obstacle};

% Storage
q_hist     = zeros(37, 3, Nsim);
% omega_hist = zeros(3, Nsim);
% u_hist     = zeros(3, Nsim);
% eq_hist    = zeros(3, Nsim);
% Psi_hist   = zeros(1, Nsim);
obstacle_hist = zeros(3, steps);

% Main simulation loop
for x = 1:steps
    % Convert degrees to Radians for computation 
    x_rad = deg2rad(x*10);
    
    % Calculate the circle's center in Cartesian coordinates
    center_x = cos(y_rad) * cos(x_rad);
    center_y = cos(y_rad) * sin(x_rad);
    % obstacles{idx.moving}  = [center_x;center_y;center_z];
    % obstacle_hist(:,x) = obstacles{idx.moving};
   
    q = [0; 0; 1];
    q = q / norm(q);
    omega = [0; 0.3; 0];
    
    % Desired equilibrium
    q_d = [0; -0.0175; -0.9998];
    q_d = q_d / norm(q_d);
    q_dot = 0;

    for k = 1:Nsim
        if k == round(Tend/2/dt) % new desired goal for t >= 5 sec
            % New desired equilibrium points
            q_d = [1; 1; 1];
            q_d = q_d / norm(q_d);
            omega_d = zeros(3,1); 
        end
    
        % === Nominal Lyapunov control ====================================
        % Attitude error function (Eq. 8)
        Psi = 2 - sqrt(2) * sqrt(1 + q' * q_d);
    
        % Configuration error vector (Eq. 12)
        e_q = (1 / (sqrt(2) * sqrt(1 + q' * q_d))) * Q' * S(q_d) * q;
    
        Psi_dot = e_q'*omega;

        % Angular velocity error
        e_w = omega - omega_d;
            
        % === Control Barrier Functions for multiple obstacles ============      
        A_all = [];
        b_all = [];

        for i = 1:length(obstacles)
            obs = obstacles{i};

            h = sin(theta_safe) - q'*obs;
            h_dot = -obs' * S(omega) * q;

            % L2fh = -obs'*S(omega)*(S(omega)*q) ...
            %      + (S(obs)*q)'*(inv(J)*( -c*omega - S(omega)*J*omega));

            L2fh = (S(omega)*obs)'*(S(omega)*q) ...
                 + (S(obs)*q)'*(J\( -c*omega - S(omega)*J*omega ));

            LgLfh = -q'*S(obs)'/J;

            % -LgLfh*u <= L2fh + k1_cbf*h_dot + k0_cbf*h (=)
            % -L2fh - LgLfh*u <= k1_cbf*h_dot + k0_cbf*h (=)
            % L2fh + LgLfh*u >= -k1_cbf*h_dot - k0_cbf*h 
            A_cbf = -LgLfh;
            b_cbf = L2fh + k1_cbf*h_dot + k0_cbf*h;

            A_all = [A_all; A_cbf];
            b_all = [b_all; b_cbf];
        end
        
        % === Control Lyapunov Functions for multiple obstacles ===========  
        % ---closed loop formula---
        % there is no CLF since a nominal actuation is given
        u_nom = -K_r*e_q - K_w*e_w + S(omega)*J*omega + c*omega;
        H = eye(3);
        f = -u_nom;

        options = optimoptions('quadprog','Display','off');
        [u,~,exitflag] = quadprog(H,f,A_all,b_all,[],[],[],[],[],options);

        % fallback if infeasible
        if exitflag ~= 1
            u = u_nom;
        end
        
        % --- Rigid body dynamics (Eq. 5)
        omega_dot = J \ (u - c*omega - S(omega)*J*omega + tau);
        omega = omega + dt * omega_dot;
    
        % --- Reduced attitude kinematics (Eq. 4)
        q_dot = S(omega) * q;
        q = q + dt * q_dot;
    
        % --- Enforce S^2 constraint
        q = q / norm(q);
    
        % --- Store
        q_hist(x,:,k)     = q;
        % omega_hist(:,k) = omega;
        % u_hist(:,k)     = u;
        % eq_hist(:,k)    = e_q;
        % Psi_hist(k)     = Psi;
    end
end

%% === Animation on S^2 ===
figure; clf; hold on; grid on; axis equal;
xlabel('x'); ylabel('y'); zlabel('z');
title('Animated Pointing Direction on S^2');
view(3);

% Draw sphere
[xs,ys,zs] = sphere(50);
surf(xs,ys,zs, ...
    'FaceAlpha',0.15,'EdgeColor','none','FaceColor',[0.8 0.8 0.8]);

for x = 1:steps
    cla;
    surf(xs,ys,zs, ...
    'FaceAlpha',0.15,'EdgeColor','none','FaceColor',[0.8 0.8 0.8]);

    % Extract trajectory
    q_hist_current = squeeze(q_hist(x,:,:));
   
    for i = 1:length(obstacles)
        % if i == 1
        %     obs = obstacle_hist(:,x);
        % else
        %     obs = obstacles{i};
        % end
        obs = obstacles{i};
        
        theta = linspace(0,2*pi,100);
        v1 = null(obs');
        v1 = v1(:,1:2);
        
        circle_local = r_safe * ...
            (v1(:,1)*cos(theta) + v1(:,2)*sin(theta));
        
        circle_points = obs + circle_local;
        circle_points = circle_points ./ vecnorm(circle_points);
        
        plot3(circle_points(1,:), circle_points(2,:), circle_points(3,:), 'LineWidth',1.5);
    end

    plot3(q_hist_current(1,1:end), q_hist_current(2,1:end), q_hist_current(3,1:end));
    plot3(q_hist_current(1,end), q_hist_current(2,end), q_hist_current(3,end));     
                 
    drawnow;
    pause(0.5);
end