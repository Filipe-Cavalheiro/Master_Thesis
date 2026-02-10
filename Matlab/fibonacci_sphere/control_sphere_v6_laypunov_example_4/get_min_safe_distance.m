function min_safe_dist = min_safe_distance(P, R, verbose)
% MIN_SAFE_DISTANCE  Computes an average minimum geodesic distance
%
%   min_safe_dist = min_safe_distance(Param, verbose)
%
%   P            : number of points (must be odd and >= 3)
%   R : sphere radius
%   verbose             : logical (true / false)

    if P < 3
        error('Not enough elements (nD must be >= 3).');
    end

    if mod(P, 2) == 0
        error('Even number of elements is not allowed.');
    end

    if nargin < 3
        verbose = false;
    end

    % Generate Fibonacci reference points
    fib_ref_points = get_fibonacci_points(P, R);
    fib_ref_points = fib_ref_points.';  % Ensure points are columns

    % Compute geodesic distances between all pairs
    ref_geo_distances = geo_distances(fib_ref_points, R);

    % Store distances excluding self-distance
    dist_vals = zeros(P, P - 1);

    for i = 1:P
        distances = ref_geo_distances(i, :);

        % Remove self-distance (zero)
        distances(i) = [];

        % Sort distances
        dist_vals(i, :) = sort(distances);
    end

    % Mean of distances for each point, then overall mean
    overall_avg_distance = mean(dist_vals(:));

    if verbose
        fprintf('overall_avg_distance: %.4f\n', overall_avg_distance);
    end

    min_safe_dist = overall_avg_distance;
end
