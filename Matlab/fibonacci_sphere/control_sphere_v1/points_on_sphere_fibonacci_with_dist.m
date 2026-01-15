clear; clc; close all;

% Number of points
P = 9;                  
N = floor((P-1)/2);         % Since P = 2N + 1 works best for odd P
K = 3; % nearest neighbor
R = 5; % Sphere radius

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
    % Cartesian coordinates of Fibonacci points
    x(i) = R * sin(theta_tmp) .* cos(phi_tmp);
    y(i) = R * sin(theta_tmp) .* sin(phi_tmp);
    z(i) = R * cos(theta_tmp);
    theta_tmp = theta_tmp + theta(i);
    phi_tmp = phi_tmp + phi;
end
x(P) = R * sin(theta_tmp) .* cos(phi_tmp);
y(P) = R * sin(theta_tmp) .* sin(phi_tmp);
z(P) = R * cos(theta_tmp);

points = [x(:), y(:), z(:)];

% Compute geodesic distances between all pairs
geo_distances_all = zeros(P, P);
for i = 1:P
    for j = 1:P
        if i ~= j
            a = points(i, :);
            b = points(j, :);
            dot_product = dot(a, b);
            geo_distances_all(i, j) = R * acos(dot_product / (R^2));
        else
            geo_distances_all(i, j) = Inf;  % ignore self-distance
        end
    end
end

% Compute geodesic distances between all pairs
cartesian_distances_all = zeros(P, P);
for i = 1:P
    for j = 1:P
        if i ~= j
            a = points(i, :);
            b = points(j, :);
            cartesian_distances_all(i, j) = sqrt(sum((a-b).^2));
        else
            cartesian_distances_all(i, j) = Inf;  % ignore self-distance
        end
    end
end


% Find 3 nearest neighbors for each point
disp('Geodesic distances to 3 nearest neighbors:');
neighbor_indices = zeros(P, K);
dist_vals = zeros(P, K);
for i = 1:P
    distances = cartesian_distances_all(i, :);
    [sorted_dists, sorted_idx] = sort(distances);
    neighbor_indices(i, :) = sorted_idx(1:K); % store neighbor indices
    dist_vals(i,:) = sorted_dists(1:K);
    fprintf('Point %d: %.4f, %.4f, %.4f\n', i-1, sorted_dists(1:K));
end

% Compute average distance per point
overall_avg_distance = mean(mean(dist_vals, 2)); % Mean across each row

% --- Plot ---
figure;
hold on;

% Draw semi-transparent sphere
[XS, YS, ZS] = sphere(100);   
surf(R*XS, R*YS, R*ZS, ...
    'FaceAlpha', 0.1, ...
    'EdgeColor', 'none', ...
    'FaceColor', [0.6 0.8 1]); 

% d2 = zeros(1, P-1);
% % Plot points
% for i = 1:P-1
%     % Compute squared distances to all other points
%     d2(i) = sqrt(sum((points(i,:) - points(i+1,:)).^2));
% end
% mean_distance = round(mean(d2), 3);

% plot3(x, y, z, '-o', ...
%     'MarkerSize', 3, ...
%     'MarkerFaceColor', 'b', ...
%     'Color', [0.1 0.2 0.8], ...
%     'LineWidth', 1);
%scatter3(x, y, z);

% Plot the points
plot3(x, y, z, 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
% Draw lines to 3 nearest neighbors
for i = 1:P
    for j = 1:K
        neighbor_idx = neighbor_indices(i, j);
        line_x = [points(i,1), points(neighbor_idx,1)];
        line_y = [points(i,2), points(neighbor_idx,2)];
        line_z = [points(i,3), points(neighbor_idx,3)];
        plot3(line_x, line_y, line_z, 'b-');
    end
end
% Formatting
axis equal;
xlabel('X'); ylabel('Y'); zlabel('Z');
grid on;
view(3);
hold off;

% % Plot histogram
% figure;
% var_theta = diff(theta);
% histogram(var_theta, 'Normalization', 'pdf')
% title("Histogram of diference between thetas")