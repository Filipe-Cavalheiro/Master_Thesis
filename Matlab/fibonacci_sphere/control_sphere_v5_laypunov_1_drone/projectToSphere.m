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
