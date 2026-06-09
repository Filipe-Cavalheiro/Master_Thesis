clc; clear; close all;

% Parameters
goal = [10;10];
obs  = [5;5];
r_obs = 1;
safety_margin = 0.5;
r_safe = r_obs + safety_margin;

alpha = 3;        % CBF gain
dt = 0.05;
T = 10;

% Initial state [x; y; theta]
x = [0;0;0];

traj = x;

for t = 0:dt:T
    
    px = x(1);
    py = x(2);
    theta = x(3);
    
    % Nominal controller (Go-to-goal)
    kp = 1;
    kr = 2;
    dx = goal(1) - px;
    dy = goal(2) - py;
    
    desired_theta = atan2(dy,dx);
    v_nom = kp * sqrt(dx^2 + dy^2);
    omega_nom = kr*(desired_theta - theta);
    
    u_nom = [v_nom; omega_nom];
    
    % CBF
    
    % Barrier function
    h = (px - obs(1))^2 + (py - obs(2))^2 - r_safe^2;
    
    % Gradient of h
    dh_dx = 2*(px - obs(1));
    dh_dy = 2*(py - obs(2));
    
    % h_dot = dh_dx * xdot + dh_dy * ydot
    % xdot = v*cos(theta)
    % ydot = v*sin(theta)
    
    Lf_h = 0; % No drift
    
    Lg_h = [dh_dx*cos(theta) + dh_dy*sin(theta), 0];
    
    % CBF inequality:
    % Lg_h * u + alpha*h >= 0
    A = -Lg_h;
    b = alpha*h;
    
    % Check constraint
    if A*u_nom <= b
        u = u_nom;
    else
        % Project onto the half-space(constraint boundary)
        u = u_nom - ((A*u_nom - b)/(A*A')) * A';
    end
    
    % Apply control
    v = u(1);
    omega = u(2);
    
    xdot = [v*cos(theta);
            v*sin(theta);
            omega];
        
    x = x + dt*xdot;
    traj = [traj x];
    
    % Stop if close to goal
    if norm(x(1:2)-goal) < 0.1
        break;
    end
end

% Plot
figure; hold on; axis equal;
plot(traj(1,:), traj(2,:), 'b','LineWidth',2);
plot(goal(1),goal(2),'gx','MarkerSize',12,'LineWidth',2);

% Obstacle
theta_c = linspace(0,2*pi,100);
plot(obs(1)+r_obs*cos(theta_c), ...
     obs(2)+r_obs*sin(theta_c),'r','LineWidth',2);

% Safety boundary
plot(obs(1)+r_safe*cos(theta_c), ...
     obs(2)+r_safe*sin(theta_c),'r--');

xlabel('X'); ylabel('Y');
title('CBF-based Obstacle Avoidance');
legend('Trajectory','Goal','Obstacle','Safety boundary');
grid on;
