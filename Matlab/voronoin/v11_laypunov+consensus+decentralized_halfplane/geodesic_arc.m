function arc_pts = geodesic_arc(p1, p2, nPoints, R, sphere_center)
    if nargin < 5
        sphere_center = [0, 0, 0]; % default to the origin if no sphere_center is provided
    end
    if nargin < 4
        R = 1; % default to unit sphere
    end

    % Translate points to the origin
    p1 = p1 - sphere_center;
    p2 = p2 - sphere_center;
    
    % Compute angular separation (omega) between the two points on the sphere
    omega = acos(dot(p1,p2) / (norm(p1) * norm(p2)));

    % Interpolation parameter (t goes from 0 to 1)
    t = linspace(0, 1, nPoints)';    
    
    % Geodesic interpolation between p1 and p2
    arc_pts = (sin((1 - t) * omega) .* p1 + sin(t * omega) .* p2) / sin(omega);
    
    % Scale the points to the sphere's radius R and translate back to the original sphere_center
    arc_pts = R * arc_pts / norm(arc_pts(1,:)); % Ensure radius R
    arc_pts = arc_pts + sphere_center; % Translate points back to the sphere's sphere_center
end