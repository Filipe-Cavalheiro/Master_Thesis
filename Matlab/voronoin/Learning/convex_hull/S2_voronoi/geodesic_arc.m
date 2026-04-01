function arc_pts = geodesic_arc(p1, p2, nPoints, R)
    if nargin < 4
        R = 1; % default unit sphere
    end

    omega = acos(dot(p1,p2)/(norm(p1)*norm(p2))); % safer with radius
    t = linspace(0,1,nPoints)';    
    arc_pts = (sin((1-t)*omega).*p1 + sin(t*omega).*p2) / sin(omega);
    arc_pts = R * arc_pts / norm(arc_pts(1,:)); % ensure radius R
end