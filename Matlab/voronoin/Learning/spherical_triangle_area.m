function A = spherical_triangle_area(a,b,c,R)
    % a, b, c: 3D points on the sphere (not necessarily unit vectors)
    % R: radius of the sphere
    % Returns the area of the spherical triangle on a sphere of radius R

    if nargin < 4
        R = 1; % default radius is 1 (unit sphere)
    end

    % normalize vectors to unit vectors to get the angles correctly
    a = a / norm(a);
    b = b / norm(b);
    c = c / norm(c);

    % angles via dot products
    alpha = acos(dot(b,c));
    beta  = acos(dot(a,c));
    gamma = acos(dot(a,b));

    s = (alpha + beta + gamma)/2;

    % L'Huilier formula (more stable than Girard theorem)
    tan_e4 = sqrt( ...
        tan(s/2)*tan((s-alpha)/2)*tan((s-beta)/2)*tan((s-gamma)/2) ...
    );

    E = 4*atan(tan_e4);  % spherical excess in radians

    A = R^2 * E;  % scale by R^2 for sphere of radius R
end