function q = randomSphericalAnnulus(p, r)
    % random angular distance
    alpha = r * (1 + rand);

    % random direction in tangent plane
    v = randn(1,3);
    v = v - dot(v,p)*p;
    v = v / norm(v);

    % Rodrigues' rotation formula
    q = cos(alpha)*p + sin(alpha)*v;
    q = q / norm(q);
end
