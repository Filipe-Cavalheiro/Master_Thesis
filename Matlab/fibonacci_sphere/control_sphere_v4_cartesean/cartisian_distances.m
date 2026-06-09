function cartesian_distances_all = cartisian_distances(points)
% Compute geodesic distances between all pairs
n_drones = size(points, 1);
cartesian_distances_all = zeros(n_drones, n_drones);
for i = 1:n_drones
    for j = 1:n_drones
        if i ~= j
            cartesian_distances_all(i, j) = sqrt(sum((points(i, :)-points(j, :)).^2));
        else
            cartesian_distances_all(i, j) = Inf;  % ignore self-distance
        end
    end
end