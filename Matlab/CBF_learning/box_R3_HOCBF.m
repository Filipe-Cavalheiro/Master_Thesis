clear; clc; close all;

% Parameters
dt = 0.01;
T = 10;
N = T/dt;

xmin = [-2.5; -2.5; -2.5];
xmax = [2.5; 2.5; 2.5];
gamma = 2.0;

% Initial state (inside box)
pos = [0; 0; 0];
pos_history = zeros(3, N);

pos_unc = pos; % same initial condition
pos_unc_history = zeros(3, N);

Kp = 3;
Ki = 0.5;
Kd = 1.0;

integral_err = zeros(3,1);
prev_err = zeros(3,1);

integral_err_unc = zeros(3,1);
prev_err_unc = zeros(3,1);

% Simulation loop
for k = 1:N
    
    % --- unconstrained --- 
    pos_ref = [2+ 5*sin(k*dt); 5*cos(k*dt); 0];
    err_unc = pos_ref - pos_unc;
    
    integral_err_unc = integral_err_unc + err_unc*dt;
    integral_err_unc = max(min(integral_err_unc, 2), -2);  % anti-windup
    
    derr = (err_unc - prev_err_unc)/dt;
    derr = max(min(derr, 10), -10);  % optional derivative limit
    
    u_nom_unc = Kp*err_unc + Ki*integral_err_unc + Kd*derr;
    
    pos_unc = pos_unc + dt*u_nom_unc;
    prev_err_unc = err_unc;

    pos_unc_history(:,k) = pos_unc;

    % --- constrained --- 
    % Error
    err = pos_ref - pos;

    % Integral
    integral_err = integral_err + err * dt;

    % Derivative
    derr = (err - prev_err) / dt;

    % PID control
    u_nom = Kp * err + Ki * integral_err + Kd * derr;

    % Store error
    prev_err = err;

    % Build CBF constraints
    [A, b] = box_cbf_constraints(pos, xmin, xmax, gamma);

    % QP: min ||u - u_nom||^2
    H = eye(3);
    f = -u_nom;

    % Solve QP
    u = quadprog(H, f, A, b, [], [], [], [], [], ...
        optimoptions('quadprog','Display','off'));

    % System update (single integrator)
    pos = pos + dt * u;

    pos_history(:,k) = pos;
end

% Plot
figure;
hold on; grid on; axis equal;

% Draw box
plot_box(xmin, xmax);

% Plot constrained trajectory
plot3(pos_history(1,:), pos_history(2,:), pos_history(3,:), ...
    'b', 'LineWidth', 2);
% Start and end points
scatter3(pos_history(1,1), pos_history(2,1), pos_history(3,1), 80, 'g', 'filled');
scatter3(pos_history(1,end), pos_history(2,end), pos_history(3,end), 80, 'r', 'filled');

% Plot unconstrained trajectory
plot3(pos_unc_history(1,:), pos_unc_history(2,:), pos_unc_history(3,:), ...
    'r', 'LineWidth', 2);
scatter3(pos_unc_history(1,1), pos_unc_history(2,1), pos_unc_history(3,1), 80, 'g', 'filled');
scatter3(pos_unc_history(1,end), pos_unc_history(2,end), pos_unc_history(3,end), 80, 'r', 'filled');

xlabel('pos'); ylabel('y'); zlabel('z');
title('CBF: State constrained inside 3D box');

view(3);
%legend({'Box','CBF Trajectory','Unconstrained','Start','End'});

% -------- FUNCTIONS --------

function [A, b] = box_cbf_constraints(pos, xmin, xmax, gamma)

h = [
    pos(1) - xmin(1);
    pos(2) - xmin(2);
    pos(3) - xmin(3);
    xmax(1) - pos(1);
    xmax(2) - pos(2);
    xmax(3) - pos(3)
];

A = [
    -1  0  0;
     0 -1  0;
     0  0 -1;
     1  0  0;
     0  1  0;
     0  0  1
];

b = gamma * h;

end

function plot_box(xmin, xmax)

% vertices
[pos_history, Y, Z] = ndgrid([xmin(1) xmax(1)], ...
                   [xmin(2) xmax(2)], ...
                   [xmin(3) xmax(3)]);
vertices = [pos_history(:), Y(:), Z(:)];

% faces
faces = [
    1 2 4 3;
    5 6 8 7;
    1 2 6 5;
    3 4 8 7;
    1 3 7 5;
    2 4 8 6
];

patch('Vertices', vertices, ...
      'Faces', faces, ...
      'FaceColor', 'none', ...
      'EdgeColor', 'k', ...
      'LineWidth', 1.5);

end