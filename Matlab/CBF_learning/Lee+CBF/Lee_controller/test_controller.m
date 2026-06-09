% 
% Copyright (c) 2020 Flight Dynamics and Control Lab
% 
% Permission is hereby granted, free of charge, to any person obtaining a
% copy of this software and associated documentation files (the 
% "Software"), to deal in the Software without restriction, including 
% without limitation the rights to use, copy, modify, merge, publish, 
% distribute, sublicense, and/or sell copies of the Software, and to permit
% persons to whom the Software is furnished to do so, subject to the 
% following conditions:
% 
% The above copyright notice and this permission notice shall be included
%  in all copies or substantial portions of the Software.
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
% OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
% MERCHANTABILITY,FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
% IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY 
% CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, 
% TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
% SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

close all; clc; clear;
addpath('aux_functions');
addpath('test_functions');

% --- Simulation parameters ---
% Model and simulation parameters
Param.Tend = 5;
Param.dTi = 0.01;  % inner-loop and simulation sampling period
Param.Nsim = round(Param.Tend/Param.dTi)+1;
Param.t = 0:Param.dTi:Param.Tend+ Param.dTi;
Param.nD = 1;
Param.gravity = 9.81;

% Quadrotor
Param.J = diag([0.02, 0.02, 0.04]);
Param.drone_mass = 2;
Param.d = 0.169;
Param.ctf = 0.0135;

% Fixed disturbance
Param.x_delta = [0.5, 0.8, -1]';
Param.R_delta = [0.2, 1.0, -0.1]';

% Other parameters
Param.g = 9.81;

% --- Controller gains ---
Param.Lee.x = 10;
Param.Lee.v = 8;
Param.Lee.i = 10;
Param.c1 = 1.5;
Param.sigma = 10;

% Attitude
Param.Lee.R = 1.5;
Param.Lee.W = 0.35;
Param.Lee.I = 10;
Param.c2 = 2;

% Yaw
Param.Lee.y = 0.8;
Param.Lee.wy = 0.15;
Param.Lee.yI = 2;
Param.c3 = 2;

% --- Initial conditions ---

% Allocate all drones memory
chaser = cell(1, Param.nD);

% Template struct
template = struct( ...
    'pos',      zeros(3, Param.Nsim), ...
    'vel',      zeros(3, Param.Nsim), ...
    'attitude', zeros(9, Param.Nsim), ...
    'ang_vel',  zeros(3, Param.Nsim), ...
    'ei',       zeros(3, Param.Nsim), ...
    'eI',       zeros(3, Param.Nsim));

[chaser{:}] = deal(template);

for iD = 1:Param.nD
    chaser{iD}.pos(:,1) = [1;1;1]; % initialize the agents on the initially desired point
    chaser{iD}.vel(:,1) = [0;0;0];
    chaser{iD}.attitude(:,1) = reshape(Euler2R([0;0;0]),[],1);
    chaser{iD}.ang_vel(:,1) = [0;0;0];
    chaser{iD}.ei(:,1) = [0;0;0];
    chaser{iD}.eI(:,1) = [0;0;0];
end

p_d_hist = zeros(3, Param.Nsim-1);
Sim.iD = 1;

target.pos = [10;10;10];
target.vel = [0;0;0];

% =========================================================
% Discrete simulation loop: x_k -> x_{k+1}
% =========================================================
for k = 1:Param.Nsim-1
    Sim.k = k;
    fprintf("Computing simulation interation: %04d\n", k);

    desired = command(chaser{1}.pos(:,k), target, Param);

    % Integrate dynamics over one sample interval
    X = struct_to_pack(k, chaser{1});
    [~, Xtmp] = ode45(@(t,XR) eom(t, XR, desired, Sim, Param), ...
                  [Param.t(k), Param.t(k+1)], ...
                  X);
    chaser{1} = pack_to_struct(k+1, chaser{1}, Xtmp(end,:));
    
    % Next state
    chaser{1} = pack_to_struct(k+1, chaser{1}, Xtmp(end,:));
end

% --- Post processing ---
% Unpack the outputs of ode45 function
x  = chaser{1}.pos;

%% Plot data
linetype = 'k';
linewidth = 1;
xlabel_ = 'time (chaser{1})';

figure;
hold on;
view(3);
plot3(x(1,:), x(2,:), x(3,:), 'k');
axis equal;
xlabel('$x_1$', 'interpreter', 'latex');
ylabel('$x_2$', 'interpreter', 'latex');
zlabel('$x_3$', 'interpreter', 'latex');
set(gca, 'Box', 'on');
grid on;
set(gca, 'FontName', 'Times New Roman');

%% Animation loop

figure;
hold on; grid on; axis equal;
view(3);

xlabel('$x_1$', 'Interpreter', 'latex');
ylabel('$x_2$', 'Interpreter', 'latex');
zlabel('$x_3$', 'Interpreter', 'latex');

set(gca, 'FontName', 'Times New Roman');

% Trajectory
plot3(x(1,:), x(2,:), x(3,:), 'k--', 'LineWidth', 1);

% Axis limits
margin = 0.5;
axis([min(x(1,:))-margin max(x(1,:))+margin ...
      min(x(2,:))-margin max(x(2,:))+margin ...
      min(x(3,:))-margin max(x(3,:))+margin]);

% Current quadrotor position
h = plot3(x(1,1), x(2,1), x(3,1), ...
    'kx', 'MarkerSize', 10, 'LineWidth', 2);

% Current desired point (diamond marker)
hd = plot3(p_d_hist(1,1), p_d_hist(2,1), p_d_hist(3,1), ...
    'rd', ...
    'MarkerSize', 8, ...
    'MarkerFaceColor', 'r', ...
    'LineWidth', 1.5);

for k = 1:Param.Nsim
    fprintf("Showing simulation interation: %04d\n", k);

    % Update quadrotor position
    set(h, ...
        'XData', x(1,k), ...
        'YData', x(2,k), ...
        'ZData', x(3,k));

    drawnow limitrate;
    pause(0.01);
end