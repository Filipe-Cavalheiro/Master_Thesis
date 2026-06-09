clear; clc; close all;

% Parameters
R = 5; % Sphere radius

% Generate first random point
phi1 = 2*pi*rand();
theta1 = acos(2*rand() - 1);

x1 = R * sin(theta1) * cos(phi1);
y1 = R * sin(theta1) * sin(phi1);
z1 = R * cos(theta1);

% phi1 = 3.2550;
% theta1 = 0.4795;
% x1 = -2.2917;
% y1 = -0.2611;
% z1 = 4.4362;

%Generate second random point
phi2 = 2*pi*rand();
theta2 = acos(2*rand() - 1);

x2 = R * sin(theta2) * cos(phi2);
y2 = R * sin(theta2) * sin(phi2);
z2 = R * cos(theta2);
% phi2 = 4.0068;
% theta2 = 0.4143;
% x2 = -1.3052;
% y2 = -1.5323;
% z2 = 4.5769;
% Distance between points
distance = sqrt((x1-x2)^2 + (y1-y2)^2 + (z1-z2)^2);

fprintf('Actual theta1: %.4f\n', theta1);
fprintf('Actual theta2: %.4f\n', theta2);

% --- Predict theta2 using linear form ---
A = (R^2 + R^2 - distance^2) / (2*R*R);  % cosine of angle between vectors
P = cos(theta1);
Q = sin(theta1) * cos(phi1 - phi2);

Rval = sqrt(P^2 + Q^2);
phiShift = atan2(Q, P);

theta2_pred1 = mod(phiShift + acos(A/Rval), pi);   % restrict to [0, pi]
theta2_pred2 = mod(phiShift - acos(A/Rval), pi);
theta2_pred3 = theta2_pred2 +pi;
theta2_pred4 = theta2_pred1 +pi;

fprintf('Pred theta2 option 1: %.4f\n', theta2_pred1);
fprintf('Pred theta2 option 2: %.4f\n', theta2_pred2);
fprintf('Pred theta3 option 1: %.4f\n', theta2_pred3);
fprintf('Pred theta4 option 2: %.4f\n', theta2_pred4);

x3 = R * sin(theta2_pred1) * cos(phi2);
y3 = R * sin(theta2_pred1) * sin(phi2);
z3 = R * cos(theta2_pred1);
x4 = R * sin(theta2_pred2) * cos(phi2);
y4 = R * sin(theta2_pred2) * sin(phi2);
z4 = R * cos(theta2_pred2);
x5 = R * sin(theta2_pred3) * cos(phi2);
y5 = R * sin(theta2_pred3) * sin(phi2);
z5 = R * cos(theta2_pred3);
x6 = R * sin(theta2_pred4) * cos(phi2);
y6 = R * sin(theta2_pred4) * sin(phi2);
z6 = R * cos(theta2_pred4);

% --- Plot ---
figure;
hold on;

% Draw semi-transparent sphere
[XS, YS, ZS] = sphere(100);   
surf(R*XS, R*YS, R*ZS, 'FaceAlpha', 0.1, 'EdgeColor', 'none', 'FaceColor', [0.6 0.8 1]);

% Plot points
plot3(x1, y1, z1, 'bo', 'MarkerSize', 6, 'MarkerFaceColor', 'b');
plot3(x2, y2, z2, 'bo', 'MarkerSize', 6, 'MarkerFaceColor', 'b');
plot3(x3, y3, z3, 'rx', 'MarkerSize', 6, 'LineWidth', 1.5);
plot3(x4, y4, z4, 'rx', 'MarkerSize', 6, 'LineWidth', 1.5);
plot3(x5, y5, z5, 'rx', 'MarkerSize', 6, 'LineWidth', 1.5);
plot3(x6, y6, z6, 'rx', 'MarkerSize', 6, 'LineWidth', 1.5);

% Draw circle along fixed phi2
theta_circle = linspace(0, pi, 200); % sweep from north to south pole
x_circle = R * sin(theta_circle) * cos(phi2);
y_circle = R * sin(theta_circle) * sin(phi2);
z_circle = R * cos(theta_circle);

plot3(x_circle, y_circle, z_circle, 'm-', 'LineWidth', 2);

% Formatting
axis equal;
xlabel('X'); ylabel('Y'); zlabel('Z');
title('Sphere with Points and Circle at Fixed \phi');
grid on;
view(3);
hold off;

dist_pred1 = norm([x1-x3, y1-y3, z1-z3]);
dist_pred2 = norm([x1-x4, y1-y4, z1-z4]);

% Select the one closer to the target distance (3m)
if abs(dist_pred1 - 3) < abs(dist_pred2 - 3)
    theta2_final = theta2_pred1;
else
    theta2_final = theta2_pred2;
end

dist_pred1 = sqrt((x1 - x3)^2 + (y1 - y3)^2 + (z1 - z3)^2);
dist_pred2 = sqrt((x1 - x4)^2 + (y1 - y4)^2 + (z1 - z4)^2);
dist_pred3 = sqrt((x1 - x5)^2 + (y1 - y5)^2 + (z1 - z5)^2);
dist_pred4 = sqrt((x1 - x6)^2 + (y1 - y6)^2 + (z1 - z6)^2);

fprintf('Distance from first point to predicted point 1: %.4f\n', dist_pred1);
fprintf('Distance from first point to predicted point 2: %.4f\n', dist_pred2);
fprintf('Distance from first point to predicted point 3: %.4f\n', dist_pred3);
fprintf('Distance from first point to predicted point 4: %.4f\n', dist_pred4);