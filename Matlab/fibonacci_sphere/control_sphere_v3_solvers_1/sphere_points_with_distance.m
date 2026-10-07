function [optimized_points, trajectory_out] = sphere_points_with_distance(N, R, D, points, K)
    % Flatten for optimization
    x0 = points(:);
    trajectory = struct('points', {}); % Initialize the trajectory

    % Optimization options
    options = optimoptions('fmincon', ...
        'Algorithm', 'sqp', ...
        'MaxFunctionEvaluations', 1e5, ...
        'OutputFcn', @(x, optimValues, state) ...
        progress_results(x, optimValues, state, N, R));
    
    % Run optimization
    x_opt = fmincon(@(x) objective(x, N), x0, [], [], [], [], [], [], ...
                    @(x) constraints(x, N, D, R, K), options);

    % Reshape optimized points
    optimized_points = reshape(x_opt, [N, 3]);
    trajectory_out = trajectory;

    % Helper: Plot results with each step as well as storing it
    function stop = progress_results(x, optimValues, state, N, R)
        persistent hFig hPlot sphereSurf
    
        stop = false;
    
        % Reshape x to N x 3
        points = reshape(x, [N, 3]);
    
        switch state
            case 'init'
                % % Setup the plot
                % hFig = figure;
                % hPlot = plot3(points(:,1), points(:,2), points(:,3), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
                % hold on;
                % 
                % % Draw unit sphere
                % [X, Y, Z] = sphere(50);
                % sphereSurf = surf(R*X, R*Y, R*Z, 'FaceAlpha', 0.1, 'EdgeColor', 'none');
                % axis equal;
                % xlabel('X'); ylabel('Y'); zlabel('Z');
                % title('Optimization Progress');
                
                
            case 'iter'
                % Update points in the plot
                % if isvalid(hPlot)
                %     set(hPlot, 'XData', points(:,1), 'YData', points(:,2), 'ZData', points(:,3));
                %     drawnow;
                %     pause(0.5); % slow down to visualize
                % end
                
                % Store this iteration’s data
                trajectory(end+1).points = points;
            case 'done'
                % Optionally, you could finalize the plot or save it
                disp('Optimization visualization done.');
        end
    end
end

%% Helper: Objective function (we're just minimizing a dummy function, i.e., 0)
function cost = objective(x, N)
    cost = 0; % We're only enforcing constraints
end

%% Helper: Constraint function
function [c, ceq] = constraints(x, N, D, R, K)
    points = reshape(x, [N, 3]);

    % Each point must lie on the sphere => norm = 1
    ceq = vecnorm(points, 2, 2) - R;

    % Inequality constraints: distance between each point and its k nearest neighbors >= D
    c = zeros(0,N);
    
    points_on_sphere = projectToSphere(points, R);

    % Compute geodesic distances between all pairs
    geo_dist = zeros(1, N);
    for i = 1:N
        a = points_on_sphere(i, :);
        for j = 1:N
            if i ~= j
                b = points_on_sphere(j, :);
                dot_product = dot(a, b);
                geo_dist(j) = R * acos(dot_product / (R^2));
            else
                geo_dist(j) = Inf;  % ignore self-distance
            end
        end
        sorted_dists = sort(geo_dist);
        nearest_dists = sorted_dists(1:K);
        c = [c; D - nearest_dists]; % ensure dist >= D => D - dist <= 0
    end
end

%% Helper: project points from R3 to an S2 sphere
function projected_points = projectToSphere(points, r)
    % projectToSphere Projects 3D Cartesian points onto a spherical surface
    %
    % Usage:
    %   projected_points = projectToSphere(points)
    %   projected_points = projectToSphere(points, r)
    %
    % Inputs:
    %   points - Nx3 matrix of (x,y,z) Cartesian coordinates
    %   r      - Radius of the sphere (default is 1)
    %
    % Output:
    %   projected_points - Nx3 matrix of points projected onto the sphere

    if nargin < 2
        r = 1;  % default radius
    end

    % Compute the norm (magnitude) of each point
    norms = sqrt(sum(points.^2, 2));

    % Avoid division by zero (optional, depending on context)
    norms(norms == 0) = eps;

    % Normalize each point and multiply by the desired radius
    projected_points = r * (points ./ norms);
end
