clear; clc; close all;

% Parameters
P = 9;                  % number of agents                 
R  = 5;                 % desired distance to target
k1 = 5; k2 = 5; k3 = 5; % controller gains
ts = 0.01;              % time step
T  = 3.5;               % total simulation time
steps = T/ts;

%Number of points for fibonacci (the number of agents MUST BE ODD)
N = floor((P-1)/2);     % Since P = 2N + 1 works best for odd P

% Generate indices
idx = -N:N;  

% Golden ratio
phi_golden = (1 + sqrt(5)) / 2;

% Fibonacci-based longitude
fibonacci_phi = 2 * pi * idx / phi_golden;

%Since we want the control to be the same for all drones
fibonacci_theta = acos(2*idx/P);
% This creates an issue, as shown in the file "points_on_sphere_mean",
% where if we have few points, the mean spacing is still very accurate.
% However, if we increase the number of points (e.g., 101),
% we get a lot of points near the top and few on the bottom of the sphere.
% This is due to the fact that points are not distribute
% uniformly in latitude.
% As a result, using the mean difference of theta (latitude) is misleading for large P,
% because it does not account for the non-uniform density of points along the sphere.

% Initial conditions (Table 1)
theta0 = [0.175 0.698 1.222 1.745 2.269 2.793 3.491 4.189 4.712];
r0     = [30 25 10 20 15 30 25 20 15];
phi0   = [1.484 1.222 1.396 1.047 1.484 1.222 0.698 1.396 0.960];

theta = theta0;
r     = r0;
phi   = phi0;

% Storage
traj = zeros(P,3,steps);  % agent positions
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
dtheta = zeros(1,P);
dphi = zeros(1,P);
theta_dot = zeros(1,P);
r_dot = zeros(1,P);
phi_dot = zeros(1,P);

% In initialization
pos = zeros(3, P); % Each column is the 3D position of an agent
for i = 1:P
    pos(:,i) = r(i) * [sin(theta(i))*cos(phi(i));
                       sin(theta(i))*sin(phi(i));
                       cos(theta(i))] + target_fun(0);
end

% Simulation loop
for k = 2:steps+1
    t = (k-1)*ts;
    
    % Target position
    po = target_fun(t);
    target(:,k) = po;
    
    p_des = po + R * [sin(fibonacci_theta).*cos(fibonacci_phi); 
                      sin(fibonacci_theta).*sin(fibonacci_phi); 
                      cos(fibonacci_theta)];

    % Compute agent positions in Cartesian coordinates
    center = mean(traj(:,:,k-1));
    for i=1:P
        pos(:,i) = center' + r(i) *[sin(theta(i))*cos(phi(i));
                          sin(theta(i))*sin(phi(i));
                          cos(theta(i))];

        v = k1 * (p_des(:,i) - pos(:,i));
        pos(:,i) = pos(:,i) + ts * v;
        traj(i,:,k) = pos(:,i)';
    end
    
    theta_dot = k1*(fibonacci_theta - theta);
    r_dot     = k2*(R*ones(1,P) - r);
    phi_dot   = k3*(fibonacci_phi - phi);
    
    % Euler integration
    theta = theta + ts*theta_dot;
    r     = r + ts*r_dot;
    phi   = phi + ts*phi_dot;
end

% 3D trajectories animation
figure; hold on; grid on; axis equal
colors = lines(P);

% Plot target path once
plot3(target(1,:), target(2,:), target(3,:), 'r', 'LineWidth', 2); 
xlabel('x'); ylabel('y'); zlabel('z');
title('3D Trajectories of 9 Agents (Case 1)');
view(3);

% Initialize plot handles for agents
endMarkers = gobjects(P,1);

% Plot the target marker
targetMarket = plot3(target(1,1), target(2,1), target(3,1), 'ro', ...
    'MarkerSize', 10, 'MarkerFaceColor', 'r');

% Create a sphere at the target location with radius R = 5
[XS, YS, ZS] = sphere(20); % 20x20 mesh grid for smooth sphere
targetSphere = surf(R*XS + target(1,1), R*YS + target(2,1), R*ZS + target(3,1), ...
    'FaceColor', 'r', 'FaceAlpha', 0.3, 'EdgeColor', 'none'); % semi-transparent

% Initialize agent markers
for i = 1:P
    endMarkers(i) = plot3(traj(i,1,1), traj(i,2,1), traj(i,3,1), 'x', ...
        'Color', colors(i,:), 'MarkerSize', 6); % end marker (will move)
end

numSteps = size(traj, 3);
for t = 1:numSteps
    % Update target marker
    set(targetMarket, 'XData', target(1,t), 'YData', target(2,t), 'ZData', target(3,t));
    
    % Update the sphere position
    set(targetSphere, 'XData', R*XS + target(1,t), ...
                      'YData', R*YS + target(2,t), ...
                      'ZData', R*ZS + target(3,t));
    
    % Update agent markers
    for i = 1:P
        set(endMarkers(i), 'XData', traj(i,1,t), 'YData', traj(i,2,t), 'ZData', traj(i,3,t));
    end
    
    drawnow;
    if t < 100
        pause(0.1); % wait 0.1 s between steps
    else
        pause(0.01);
    end
end
