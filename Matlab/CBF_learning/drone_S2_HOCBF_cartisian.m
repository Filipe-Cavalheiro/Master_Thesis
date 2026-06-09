%The purpose of this code is to avoid a barrier in S^2
clc; clear; close all;

% === Simulation parameters ===
Tend = 10;
dt   = 0.01;
Nsim = round(Tend/dt) + 1;
t    = linspace(0, Tend, Nsim);

% === Parameters CBF ===
% Parameters for the circle

y = 180;  % Latitude in degrees
r_safe = 0.1;
theta_safe = acos(1-r_safe^2/2);
y_rad = deg2rad(y);
center_z = sin(y_rad);

k0 = 3;     % CBF gain
k1 = 3;     % CBF gain

% System parameters (V. Numerical Simulations)
c = 0.3;                             % Nm (s/rad)
J = diag([0.0294 0.0305 0.0495]);    % kg m^2
tau = zeros(3,1);

% Controller gains (29a, 29b)
K_r = diag([7 7 7]);
K_w = diag([7.056 7.320 11.88]) * 1e-1;

% Hat (skew) operator
S = @(x) [   0   -x(3)  x(2);
           x(3)    0   -x(1);
          -x(2)  x(1)    0 ];

% Initial conditions
Q = eye(3);
omega_d = zeros(3,1);

steps = length(0:10:360);

% Storage
q_hist     = zeros(37, 3, Nsim);
% omega_hist = zeros(3, Nsim);
% u_hist     = zeros(3, Nsim);
% eq_hist    = zeros(3, Nsim);
% Psi_hist   = zeros(1, Nsim);

% Main simulation loop
for x = 1:steps
    % Convert degrees to Radians for computation 
    x_rad = deg2rad(x*10);
    
    % Calculate the circle's center in Cartesian coordinates
    center_x = cos(y_rad) * cos(x_rad);
    center_y = cos(y_rad) * sin(x_rad);

    obstacle  = [center_x;center_y;center_z];

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
    
        % Angular velocity error
        e_w = omega - omega_d;
    
        % Control law (Eq. 28, reduced to S^2)
        u_nom = Q'* (-K_r * Q * e_q ...
            -K_w * e_w )...
            + S(omega) * J * omega ...
            + c * omega ...
            - tau;
        
        % === Control barrier function=====================================
        h =  r_safe^2 - norm(q - obstacle)^2;
        h_dot = 2*(q - obstacle)' * (S(omega)*q);
       
        Lfh = 2*(q - obstacle)' * (S(omega)*q);
        Lgh = 0;
        
        % L^2fg
        L2gh = 0;
        % L^2fh
        L2fh = (2*S(omega)*obstacle)'*S(omega)*q+2*(S(q)*obstacle)'*inv(J)*(-c*omega - S(omega)*J*omega);
        %LgLfh
        LgLfh = 2*(S(q)*obstacle)'/J;
        % Control effectiveness term
    
        A = LgLfh;
        b = -L2fh - k1*h_dot - k0*h;
        % Check constraint
        if A*u_nom <= b
            u = u_nom;
        else
            % Project onto the half-space(constraint boundary)
            u = u_nom - ((A*u_nom - b)/(A*A')) * A';
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


for x = 1:steps-1
    cla;
    surf(xs,ys,zs, ...
    'FaceAlpha',0.15,'EdgeColor','none','FaceColor',[0.8 0.8 0.8]);

    % Get obstacle for this longitude step
    x_rad = deg2rad(x*10);
    center_x = cos(y_rad)*cos(x_rad);
    center_y = cos(y_rad)*sin(x_rad);
    obstacle  = [center_x; center_y; center_z];
    
    % Extract trajectory
    q_hist_current = squeeze(q_hist(x,:,:));
    Nsim = size(q_hist_current,2);
    
    % --- Compute safety circle ---
    theta = linspace(0,2*pi,100);
    v1 = null(obstacle');
    v1 = v1(:,1:2);
    
    circle_local = r_safe * ...
        (v1(:,1)*cos(theta) + v1(:,2)*sin(theta));
    
    circle_points = obstacle + circle_local;
    circle_points = circle_points ./ vecnorm(circle_points);
   
    plot3(q_hist_current(1,1:Nsim), ...
          q_hist_current(2,1:Nsim), ...
          q_hist_current(3,1:Nsim), 'b', 'LineWidth',1.5);
    
    plot3(q_hist_current(1,Nsim), ...
          q_hist_current(2,Nsim), ...
          q_hist_current(3,Nsim), 'ro', 'MarkerSize',6,'MarkerFaceColor','r'); 
        plot3(circle_points(1,:), circle_points(2,:), circle_points(3,:))
                     
    drawnow;
    pause(0.5);
end