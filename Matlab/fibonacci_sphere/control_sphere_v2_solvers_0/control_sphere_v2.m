clear; clc; close all;

% Number of fibonacci_points
P = 9;                  
N = floor((P-1)/2);     % Since P = 2N + 1 works best for odd P
K = 3;                  % nearest neighbor
R = 5;                  % Sphere radius

% Golden ratio
phi_golden = (1 + sqrt(5)) / 2;

% Generate indices
idx = -N:N;

% Latitude and phi
theta_pos = acos(2*idx/P);      % latitude
theta = diff(theta_pos);
phi_pos = 2*pi*idx/phi_golden;  % longitude
phi = diff(phi_pos);
phi = phi(1) ;

theta_tmp = theta_pos(1);
phi_tmp = phi_pos(1);

x = zeros(1,P-1);
y = zeros(1,P-1);
z = zeros(1,P-1);

for i = 1:P-1
    % Cartesian coordinates of Fibonacci fibonacci_points
    x(i) = R * sin(theta_tmp) .* cos(phi_tmp);
    y(i) = R * sin(theta_tmp) .* sin(phi_tmp);
    z(i) = R * cos(theta_tmp);
    theta_tmp = theta_tmp + theta(i);
    phi_tmp = phi_tmp + phi;
end
x(P) = R * sin(theta_tmp) .* cos(phi_tmp);
y(P) = R * sin(theta_tmp) .* sin(phi_tmp);
z(P) = R * cos(theta_tmp);

fibonacci_points = [x(:), y(:), z(:)];

% Compute geodesic distances between all pairs
geo_distances_all = zeros(P, P);
for i = 1:P
    for j = 1:P
        if i ~= j
            a = fibonacci_points(i, :);
            b = fibonacci_points(j, :);
            dot_product = dot(a, b);
            geo_distances_all(i, j) = R * acos(dot_product / (R^2));
        else
            geo_distances_all(i, j) = Inf;  % ignore self-distance
        end
    end
end

% Compute geodesic distances between all pairs
% cartesian_distances_all = zeros(P, P);
% for i = 1:P
%     for j = 1:P
%         if i ~= j
%             a = fibonacci_points(i, :);
%             b = fibonacci_points(j, :);
%             cartesian_distances_all(i, j) = sqrt(sum((a-b).^2));
%         else
%             cartesian_distances_all(i, j) = Inf;  % ignore self-distance
%         end
%     end
% end

% Find 3 nearest neighbors for each point
%disp('Geodesic distances to K nearest neighbors:');
if K <= P
    K = P-1;
end
neighbor_indices = zeros(P, K);
dist_vals = zeros(P, K);

for i = 1:P
    distances = geo_distances_all(i, :);
    [sorted_dists, sorted_idx] = sort(distances);
    neighbor_indices(i, :) = sorted_idx(1:K); % store neighbor indices
    dist_vals(i,:) = sorted_dists(1:K);
    %fprintf('Point %d: %.4f, %.4f, %.4f\n', i-1, sorted_dists(1:K));
end

% Compute average distance per point
overall_avg_distance = mean(mean(dist_vals, 2)); % Mean across each row
fprintf('overall_avg_distance: %.4f \n', overall_avg_distance);

theta0 = [0.175 0.698 1.222 1.745 2.269 2.793 3.491 4.189 4.712];
r0     = [30 25 10 20 15 30 25 20 15];
phi0   = [1.484 1.222 1.396 1.047 1.484 1.222 0.698 1.396 0.960];

points = r0.* [sin(theta0).*cos(phi0);
                sin(theta0).*sin(phi0);
                cos(theta0)];

points = points';
if P ~= 9 
    points = 15*randn(P, 3);
end 
optimized_points = sphere_points_with_distance(P, R, overall_avg_distance, points, K);

% Display result
figure;
plot3(optimized_points(:,1), optimized_points(:,2), optimized_points(:,3), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
hold on;
[X,Y,Z] = sphere(50);
surf(R*X,R*Y,R*Z, 'FaceAlpha', 0.1, 'EdgeColor', 'none');
axis equal;
title('Optimized Points on Sphere with Safe Distance D');
xlabel('X'); ylabel('Y'); zlabel('Z');

% Compute geodesic distances between all pairs
cartesian_distances_all = zeros(P, P);
for i = 1:P
    for j = 1:P
        if i ~= j
            a = optimized_points(i, :);
            b = optimized_points(j, :);
            cartesian_distances_all(i, j) = sqrt(sum((a-b).^2));
        else
            cartesian_distances_all(i, j) = Inf;  % ignore self-distance
        end
    end
end

% Find K nearest neighbors for each point
%disp('Geodesic distances to K nearest neighbors:');
for i = 1:P
    distances = cartesian_distances_all(i, :);
    [sorted_dists, sorted_idx] = sort(distances);
    fprintf('Point %d: %.4f, %.4f, %.4f\n\n', i-1, sorted_dists(1:K));
end

% % --- Plot ---
% figure;
% hold on;
% 
% % Draw semi-transparent sphere
% [XS, YS, ZS] = sphere(100);   
% surf(R*XS, R*YS, R*ZS, ...
%     'FaceAlpha', 0.1, ...
%     'EdgeColor', 'none', ...
%     'FaceColor', [0.6 0.8 1]); 
%
% d2 = zeros(1, P-1);
% % Plot fibonacci_points
% for i = 1:P-1
%     % Compute squared distances to all other fibonacci_points
%     d2(i) = sqrt(sum((fibonacci_points(i,:) - fibonacci_points(i+1,:)).^2));
% end
% mean_distance = round(mean(d2), 3);
%
% plot3(x, y, z, '-o', ...
%     'MarkerSize', 3, ...
%     'MarkerFaceColor', 'b', ...
%     'Color', [0.1 0.2 0.8], ...
%     'LineWidth', 1);
%scatter3(x, y, z);
%
% Plot the fibonacci_points
%plot3(x, y, z, 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
% Draw lines to 3 nearest neighbors
% for i = 1:P
%     for j = 1:K
%         neighbor_idx = neighbor_indices(i, j);
%         line_x = [fibonacci_points(i,1), fibonacci_points(neighbor_idx,1)];
%         line_y = [fibonacci_points(i,2), fibonacci_points(neighbor_idx,2)];
%         line_z = [fibonacci_points(i,3), fibonacci_points(neighbor_idx,3)];
%         plot3(line_x, line_y, line_z, 'b-');
%     end
% end
% Formatting
% axis equal;
% xlabel('X'); ylabel('Y'); zlabel('Z');
% grid on;
% view(3);
% hold off;
%
% % Plot histogram
% figure;
% var_theta = diff(theta);
% histogram(var_theta, 'Normalization', 'pdf')
% title("Histogram of diference between thetas")
