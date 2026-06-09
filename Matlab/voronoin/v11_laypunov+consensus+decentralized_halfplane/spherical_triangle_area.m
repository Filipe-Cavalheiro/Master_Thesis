function A = spherical_triangle_area(a,b,c)
    % angles via dot products
    alpha = acos(dot(b,c));
    beta  = acos(dot(a,c));
    gamma = acos(dot(a,b));

    s = (alpha + beta + gamma)/2;

    % L'Huilier formula (more stable)
    tan_e4 = sqrt( ...
        tan(s/2)*tan((s-alpha)/2)*tan((s-beta)/2)*tan((s-gamma)/2) ...
    );

    A = 4*atan(tan_e4);
end