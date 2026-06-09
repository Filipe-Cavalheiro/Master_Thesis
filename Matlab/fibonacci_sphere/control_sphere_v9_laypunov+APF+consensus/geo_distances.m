function geo_distances_all = geo_distances(points, R, sphere_center)
% Compute geodesic distances between all pairs

    if nargin < 3 || isempty(sphere_center)
        sphere_center = [0;0;0];
    end

    n_drones = size(points, 2);
    geo_distances_all = zeros(n_drones, n_drones);
    for i = 1:n_drones
        for j = 1:n_drones
            if i ~= j
                a = points(:, i);
                b = points(:, j);
                dot_product = dot(a-sphere_center, b-sphere_center);
                geo_distances_all(i, j) = R * acos(dot_product / (R^2));
            else
                geo_distances_all(i, j) = Inf;  % ignore self-distance
            end
        end
    end
end