clear; clc; close all;

% Number of points
P = 1001;                  
N = floor((P-1)/2);         % Since P = 2N + 1 works best for odd P
K = 5; % nearest neighbor

% Sphere radius
R = 5;

% Golden ratio
phi_golden = (1 + sqrt(5)) / 2;

% Generate indices
idx = -N:N; 
% Fibonacci-based longitude
fibonacci_phi = 2*pi*idx/phi_golden;

theta = linspace(0,pi,P); 

% Compute mean angular spacing (now uniform)
theta_mean = pi/P; % latitude
phi_mean = mean(diff(fibonacci_phi)); %longitude

x = zeros(1, P-1);
y = zeros(1, P-1);
z = zeros(1, P-1);
%z = linspace(-R, R, P);
% Cartesian coordinates of Fibonacci points
for i = 1:P
    x(i) = R * sin(i*theta_mean)* cos(i*phi_mean);
    y(i) = R * sin(i*theta_mean)* sin(i*phi_mean);
    z(i) = R * cos(i*theta_mean);
end 

points = [x(:), y(:), z(:)];

% --- Plot ---
figure;
hold on;
% Draw semi-transparent sphere
[XS, YS, ZS] = sphere(100);   
surf(R*XS, R*YS, R*ZS, ...
    'FaceAlpha', 1.0, ...
    'EdgeColor', 'none', ...
    'FaceColor', [0.7647 0.7647 0.7647]); 

% plot3(x, y, z, '-o', ...
%     'MarkerSize', 3, ...
%     'MarkerFaceColor', 'b', ...
%     'Color', [0.1 0.2 0.8], ...
%     'LineWidth', 1);
scatter3(x, y, z, 15, 'k', 'filled')
% Formatting for paper
axis equal
axis off
set(gca, 'Visible', 'off')
set(gcf, 'Color', 'w');          % white background
set(gca, 'Position', [0 0 1 1]); % fill figure
view(0,90);

exportgraphics(gcf, 'fibonacci_sphere_mean_theta_top.png', 'ContentType', 'vector');
hold off;

% Plot histogram
figure;
x = diff(theta);
histogram(x, 'Normalization', 'pdf')
title("Histogram of diference between thetas")
