function q = AFP_safe_guard(iD,  current_pos, q, min_safe_dist, Rad, c_sphere, Param)
%AFP_SAFE_GUARD Summary of this function goes here
%   Detailed explanation goes here
% parse the position from cells to array

    % Project points on the sphere
    points_on_sphere = projectToSphere(current_pos, Rad, c_sphere);

    %Compute geodesic distances between all pairs
    current_geo_distances = geo_distances(points_on_sphere, Rad, c_sphere);

    % APF collision avoidance
    distances = current_geo_distances(iD, :);
    [sorted_dists, sorted_idx] = sort(distances);
    dist_vals = sorted_dists(1:(Param.nD-1));

    artificial_potential_vector = zeros(3,1);
    for i = 1:Param.nD-1
        if dist_vals(i) > min_safe_dist
            continue;
        end
        A_point = points_on_sphere(:, iD);
        B_point = points_on_sphere(:, sorted_idx(i));

        nA = (A_point - c_sphere);
        nA = nA / norm(nA);

        AB = A_point - B_point;

        % Remove radial component
        AB_tangent = AB - dot(AB, nA) * nA;

        if norm(AB_tangent) < 1e-6
            continue;
        end

        AB_tangent = AB_tangent / norm(AB_tangent);

        repulsion_mag = (min_safe_dist - dist_vals(i))* 0.5;

        artificial_potential_vector = ...
            artificial_potential_vector + repulsion_mag * AB_tangent;
    end

    % Clamp
    if norm(artificial_potential_vector) > min_safe_dist
        artificial_potential_vector = ...
            artificial_potential_vector / norm(artificial_potential_vector) * min_safe_dist;
    end

    % Apply as displacement
    q = points_on_sphere(:,iD) + artificial_potential_vector;
    q = projectToSphere(q, Rad, c_sphere);
end

