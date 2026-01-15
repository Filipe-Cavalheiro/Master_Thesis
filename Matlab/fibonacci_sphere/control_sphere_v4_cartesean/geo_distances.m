function geo_distances_all = geo_distances(points, R)
% Compute geodesic distances between all pairs
n_drones = size(points, 1);
geo_distances_all = zeros(n_drones, n_drones);
for i = 1:n_drones
    for j = 1:n_drones
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