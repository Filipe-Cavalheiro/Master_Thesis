function [ip, it] = cart2sphGrid(p, nTheta, nPhi)
    [theta,phi,~] = cart2sph(p(1),p(2),p(3));
    theta = mod(theta,2*pi);
    phi = pi/2 - phi;

    it = floor(theta/(2*pi) * nTheta) + 1;
    ip = floor(phi/pi * nPhi) + 1;
end
