function A = spherical_triangle_area(a,b,c)
    % a, b, c: 3D points on the sphere (not necessarily unit vectors)
    % R: radius of the sphere
    % Returns the area of the spherical triangle on a sphere of radius R

    % angles via dot products
    alpha = acos(dot(b,c));
    beta  = acos(dot(a,c));
    gamma = acos(dot(a,b));

    s = (alpha + beta + gamma)/2;

    % L'Huilier formula (more stable than Girard theorem)
    tan_e4 = sqrt( ...
        tan(s/2)*tan((s-alpha)/2)*tan((s-beta)/2)*tan((s-gamma)/2) ...
    );

    A = 4*atan(tan_e4);  % spherical excess in radians
end