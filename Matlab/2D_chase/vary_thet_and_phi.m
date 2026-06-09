% Cooperative Control for Target Capturing Task
% Case 1 from Kim & Sugie (2007), Automatica 43, pp.1426–1431
% 9 agents pursuing a moving target in 3D using cyclic pursuit strategy.

clear; clc; close all;
% Parameters
n  = 9;                 % number of agents
R  = 5;                 % desired distance to target
k1 = 5; k2 = 5; k3 = 5; % controller gains
ts = 0.01;              % time step
T  = 5;                 % total simulation time
steps = T/ts;

phi_des = 0;
theta_des = 0;

% Initial conditions (Table 1)
theta0 = [0.175 0.698 1.222 1.745 2.269 2.793 3.491 4.189 4.712];
r0     = [30 25 10 20 15 30 25 20 15];
phi0   = [1.484 1.222 1.396 1.047 1.484 1.222 0.698 1.396 0.960];

theta = theta0; 
r     = r0;
phi   = phi0;

% Storage
traj = zeros(n,3,steps);  % agent positions
target = zeros(3,steps);

% Target trajectory (from paper)
% po(t) = [3t,0,0]T for 0<=t<=1.5
%       = [3t, -3(t-1.5), 0]T for 1.5<t<=3
%       = [9,-4.5,0]T for 3<t<=5
target_fun = @(t) [ ...
    (t<=1.5).*3*t + (t>1.5 & t<=3).*3*t + (t>3).*9 ; ...
    (t<=1.5).*0    + (t>1.5 & t<=3).*(-3*(t-1.5)) + (t>3).*(-4.5) ; ...
    zeros(size(t)) ];

% Initialize control laws vars
dtheta = zeros(1,n);
dphi = zeros(1,n);

% Simulation loop
for k = 1:steps
    t = (k-1)*ts;
    
    % Target position
    po = target_fun(t);
    target(:,k) = po;
    
    % Compute agent positions in Cartesian coordinates
    for i=1:n
        xi = (r(i)/2)*(cos(theta(i))+cos(phi(i)));
        yi = r(i)*sin(theta(i));
        zi = r(i)*sin(phi(i));
        traj(i,:,k) = [xi, yi, zi] + po';
    end

    % Control laws (update polar coords)
    for i=1:n-1
        dphi(i) = phi(i+1) - phi(i);
    end
    %last one is done outside of the loop
    dphi(n) =   phi(1) - phi(n) + 2*pi;

    for i=1:n-1
        dtheta(i) = theta(i+1) - theta(i);
    end
    %last one is done outside of the loop
    dtheta(n) = theta(1) - theta(n) + 2*pi;
    
    theta_dot = k1*dtheta;
    r_dot     = k2*(R - r);
    phi_dot   = k3*dphi;
    
    % Euler integration
    theta = theta + ts*theta_dot;
    r     = r + ts*r_dot;
    phi   = phi + ts*phi_dot;
end

% Plot results
% 3D trajectories
figure; hold on; grid on; axis equal
colors = lines(n);
for i=1:n
    plot3(squeeze(traj(i,1,:)), squeeze(traj(i,2,:)), squeeze(traj(i,3,:)), ...
        'Color', 'k', 'LineWidth',0.5);
    plot3(traj(i,1,1), traj(i,2,1), traj(i,3,1), 'o', 'Color', colors(i,:), 'MarkerSize',6); % start
    plot3(traj(i,1,end), traj(i,2,end), traj(i,3,end), 'o', 'Color', colors(i,:), 'MarkerSize',6); % end
end
view(3);
plot3(target(1,:),target(2,:),target(3,:),'r','LineWidth',2); % target path
xlabel('x'); ylabel('y'); zlabel('z');
title('3D Trajectories of 9 Agents (Case 1)');
hold off;

% XY projection
% figure; hold on; grid on; axis equal
% for i=1:n
%     plot(squeeze(traj(i,1,:)), squeeze(traj(i,2,:)), 'Color', colors(i,:), 'LineWidth',1.5);
%     plot(traj(i,1,1), traj(i,2,1), 'o', 'Color', colors(i,:));
%     plot(traj(i,1,end), traj(i,2,end), 's', 'Color', colors(i,:));
% end
% plot(target(1,:),target(2,:),'r','LineWidth',2);
% xlabel('x'); ylabel('y');
% title('XY Projection of Agent Trajectories (Case 1)');
