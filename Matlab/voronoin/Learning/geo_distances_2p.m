function geo_distance = geo_distances_2p(p1, p2, R, sphere_center)
% Compute geodesic distances between all pairs

    if nargin < 3 || isempty(R)
        R = 1;
    end

    if nargin < 4 || isempty(sphere_center)
        sphere_center = [0;0;0];
    end

    cos_theta = dot(p1-sphere_center, p2-sphere_center) / (R^2);
    geo_distance = R * acos(cos_theta);
end