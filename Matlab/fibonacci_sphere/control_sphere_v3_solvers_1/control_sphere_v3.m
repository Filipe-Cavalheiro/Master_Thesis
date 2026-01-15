function trajectory = control_sphere_v3(P, R, inital_pos)
    % This function returns the desired trajectory for the drones to form a 
    % sphere around a point
    % input parameters:
    % P - number of drones
    % R - radios of sphere

    % Number of fibonacci_points           
    N = floor((P-1)/2);     % Since P = 2N + 1 works best for odd P
    K = 3;                  % nearest neighbor
    
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
    
    x_fibonacci = zeros(1,P-1);
    y_fibonacci = zeros(1,P-1);
    z_fibonacci = zeros(1,P-1);
    
    for i = 1:P-1
        % Cartesian coordinates of Fibonacci fibonacci_points
        x_fibonacci(i) = R * sin(theta_tmp) .* cos(phi_tmp);
        y_fibonacci(i) = R * sin(theta_tmp) .* sin(phi_tmp);
        z_fibonacci(i) = R * cos(theta_tmp);
        theta_tmp = theta_tmp + theta(i);
        phi_tmp = phi_tmp + phi;
    end
    x_fibonacci(P) = R * sin(theta_tmp) .* cos(phi_tmp);
    y_fibonacci(P) = R * sin(theta_tmp) .* sin(phi_tmp);
    z_fibonacci(P) = R * cos(theta_tmp);
    
    fibonacci_points = [x_fibonacci(:), y_fibonacci(:), z_fibonacci(:)];
    
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
    
    % theta0 = [0.175 0.698 1.222 1.745 2.269 2.793 3.491 4.189 4.712];
    % r0     = [30 25 10 20 15 30 25 20 15];
    % phi0   = [1.484 1.222 1.396 1.047 1.484 1.222 0.698 1.396 0.960];
    % 
    % points = r0.* [sin(theta0).*cos(phi0);
    %                 sin(theta0).*sin(phi0);
    %                 cos(theta0)];
    
    points = inital_pos';
    [optimized_points, trajectory] = sphere_points_with_distance(P, R, overall_avg_distance, points, K);
    
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
        fprintf('Point %d: %.4f, %.4f\n\n', i-1, sorted_dists(1:K));
    end
    
    % % --- Plot ---
    % 3D trajectories
    figure; hold on; grid on; axis equal
    colors = lines(P);
    for i=1:P
        drone_traj = cell2mat(arrayfun(@(s) s.points(i,:), trajectory, 'UniformOutput', false)');
        plot3(squeeze(drone_traj(:, 1)), squeeze(drone_traj(:, 2)), squeeze(drone_traj(:, 3)), ...
            'Color', colors(i,:), 'LineWidth',0.5);
        plot3(drone_traj(1, 1), drone_traj(1, 2), drone_traj(1, 3), 'o', 'Color', colors(i,:), 'MarkerSize',6); % start
        plot3(optimized_points(:, 1), optimized_points(:, 2), optimized_points(:, 3), 'x', 'Color', colors(i,:), 'MarkerSize',6); % end
    end
    view(3);
    xlabel('x'); ylabel('y'); zlabel('z');
    title('3D Trajectories of 9 Agents (Case 1)');
    hold off;
end