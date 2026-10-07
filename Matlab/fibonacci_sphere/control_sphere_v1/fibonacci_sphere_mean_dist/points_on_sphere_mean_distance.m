clear; clc; close all;

% Parameters
P = 1001;                  % number of agents
drone_r  = 5;           % desired distance to target
k1 = 5; k2 = 5; k3 = 5; % controller gains
ts = 0.01;              % time step
T  = 3.5;               % total simulation time
steps = T/ts;

% Golden ratio
phi_golden = (1 + sqrt(5)) / 2;

% Generate indices
N = floor((P-1)/2);         % Since P = 2N + 1 works best for odd P
idx = -N:N;

% Latitude and longitude
theta_fibonacci  = asin(2*idx/P);          % latitude
phi_fibonacci = 2*pi*idx/phi_golden;       % longitude

% Cartesian coordinates of Fibonacci points
x_fibonacci = drone_r * sin(theta_fibonacci) .* cos(phi_fibonacci);
y_fibonacci = drone_r * sin(theta_fibonacci) .* sin(phi_fibonacci);
z_fibonacci = drone_r * cos(theta_fibonacci);

% % --- Plot ---
% figure;
% hold on;
% 
% % Draw semi-transparent sphere
% [XS, YS, ZS] = sphere(100);   
% surf(drone_r*XS, drone_r*YS, drone_r*ZS, ...
%     'FaceAlpha', 0.1, ...
%     'EdgeColor', 'none', ...
%     'FaceColor', [0.6 0.8 1]); 
%
% plot3(x_fibonacci, y_fibonacci, z_fibonacci, '-o', ...
%     'MarkerSize', 3, ...
%     'MarkerFaceColor', 'b', ...
%     'Color', [0.1 0.2 0.8], ...
%     'LineWidth', 1);
% % Formatting
% axis equal;
% xlabel('X'); ylabel('Y'); zlabel('Z');
% title('Fibonacci Lattice: 1000 points with 3-nearest neighbor connections');
% grid on;
% view(3);
% hold off;

% % Plot histogram
% figure;
% x = diff(theta_fibonacci);
% histogram(x, 'Normalization', 'pdf')
% title("Histogram of diference between thetas")

points_fibonacci = [x_fibonacci(:), y_fibonacci(:), z_fibonacci(:)];
d2_fibonacci = zeros(1, P-1);
% Plot points
for i = 1:P-1
    % Compute squared distances to all other points
    d2_fibonacci(i) = sqrt(sum((points_fibonacci(i,:) - points_fibonacci(i+1,:)) .^ 2));
end
mean_distance = round(mean(d2_fibonacci), 3);
r1 = drone_r;
r2 = drone_r;
desired_theta = zeros(1,P);
% --- Predict phi2 using linear form ---
% Cosine of angle between vectors

desired_theta(1) = theta_fibonacci(1);
x = zeros(1,P);
y = zeros(1,P);
z = zeros(1,P);
x(1) = drone_r * sin(desired_theta(1)) .* cos(phi_fibonacci(1));
y(1) = drone_r * sin(desired_theta(1)) .* sin(phi_fibonacci(1));
z(1) = drone_r * cos(desired_theta(1));
for i = 2:floor(P/2)+1
    A = (r1^2 + r2^2 - mean_distance^2) / (2*r1*r2);
    % Coefficients
    var1 = cos(desired_theta(i-1));
    var2 = sin(desired_theta(i-1))*cos(phi_fibonacci(i-1) - phi_fibonacci(i));
    % Solve equation: P cos(phi2) + Q sin(phi2) = A
    Rval = sqrt(var1^2 + var2^2);
    phiShift = atan2(var2, var1);
    opt1 = mod(phiShift + acos(A/Rval), pi);
    opt2 = mod(phiShift - acos(A/Rval), pi);
    opt3 = opt1+pi;
    opt4 = opt2+pi;
    x1_tmp = drone_r * sin(opt1) .* cos(phi_fibonacci(i));
    y1_tmp = drone_r * sin(opt1) .* sin(phi_fibonacci(i));
    z1_tmp = drone_r * cos(opt1);
    x2_tmp = drone_r * sin(opt2) .* cos(phi_fibonacci(i));
    y2_tmp = drone_r * sin(opt2) .* sin(phi_fibonacci(i));
    z2_tmp = drone_r * cos(opt2);
    x3_tmp = drone_r * sin(opt3) .* cos(phi_fibonacci(i));
    y3_tmp = drone_r * sin(opt3) .* sin(phi_fibonacci(i));
    z3_tmp = drone_r * cos(opt3);
    x4_tmp = drone_r * sin(opt4) .* cos(phi_fibonacci(i));
    y4_tmp = drone_r * sin(opt4) .* sin(phi_fibonacci(i));
    z4_tmp = drone_r * cos(opt4);
    d1 = norm([x(i-1) - x1_tmp,y(i-1)- y1_tmp,z(i-1)- z1_tmp]);
    d2 = norm([x(i-1) - x2_tmp,y(i-1)- y2_tmp,z(i-1)- z2_tmp]);
    d3 = norm([x(i-1) - x3_tmp,y(i-1)- y3_tmp,z(i-1)- z3_tmp]);
    d4 = norm([x(i-1) - x4_tmp,y(i-1)- y4_tmp,z(i-1)- z4_tmp]);
    distances = [d1, opt1; d2, opt2; d3, opt3; d4, opt4];
    distances = [round(distances(:,1), 3),distances(:,2)];
    distances = distances(abs(distances(:,1) - mean_distance) < 1e-6, :);
    % filter rows where second column > desired_theta
    %distances = distances(distances(:,2) < desired_theta(i-1), :);
    % pick the one with the smallest second column
    [~, idx] = min(distances(:,2));
    result = distances(idx, :);
    x(i) = drone_r * sin(result(2)) .* cos(phi_fibonacci(i));
    y(i) = drone_r * sin(result(2)) .* sin(phi_fibonacci(i));
    z(i) = drone_r * cos(result(2));
    desired_theta(i) = result(2);
    fprintf('Pred phi2 option 2: %.4f\n', desired_theta(i));
end
x =  [x(1:floor(P/2)+1), flip(x(1:floor(P/2)))];
y = [y(1:floor(P/2)+1), -flip(y(1:floor(P/2)))];
z = [z(1:floor(P/2)+1), -flip(z(1:floor(P/2)))];

points = [x(:), y(:), z(:)];
% Plot points
d2 = zeros(1, P-1);
for i = 1:P-1
    % Compute squared distances to all other points
    d2(i) = sqrt(sum((points(i,:) - points(i+1,:)) .^ 2));
end

% --- Plot ---
figure;
hold on;
% Draw semi-transparent sphere
[XS, YS, ZS] = sphere(100);   
surf(drone_r*XS, drone_r*YS, drone_r*ZS, ...
    'FaceAlpha', 1.0, ...
    'EdgeColor', 'none', ...
    'FaceColor', [0.7647 0.7647 0.7647]); 

% plot3(x, y, z, '-o', ...
%     'MarkerSize', 3, ...
%     'MarkerFaceColor', 'b', ...
%     'Color', [0.1 0.2 0.8], ...
%     'LineWidth', 1);
scatter3(x, y, z, 15, 'k', 'filled')

scatter3(x, y, z, 15, 'k', 'filled')
% Formatting for paper
axis equal
axis off
set(gca, 'Visible', 'off')
set(gcf, 'Color', 'w');          % white background
set(gca, 'Position', [0 0 1 1]); % fill figure
view(3)
hold off;


print2pdf("fibonacci_sphere_mean_dist_normal", true)