function points = poissonDiskSampling(r, k)
% Bridson Poisson Disk Sampling on the unit sphere
% r : minimum angular distance (radians)
% k : candidates per active point (default 30)

if nargin < 2
    k = 30;
end

cellSize = r / sqrt(2);
nTheta = ceil(2*pi / cellSize);
nPhi   = ceil(pi / cellSize);

grid = cell(nPhi, nTheta);
points = [];
activeList = [];

% initial point
p0 = randomPointOnSphere();
points(end+1,:) = p0;
activeList(end+1,:) = p0;

[phi0, theta0] = cart2sphGrid(p0, nTheta, nPhi);
grid{phi0, theta0} = p0;

while ~isempty(activeList)
    idx = randi(size(activeList,1));
    p = activeList(idx,:);
    found = false;

    for i = 1:k
        q = randomSphericalAnnulus(p, r);

        [phi, theta] = cart2sphGrid(q, nTheta, nPhi);

        ok = true;
        for ip = max(phi-2,1):min(phi+2,nPhi)
            for dTheta = -2:2
                it = mod(theta - 1 + dTheta, nTheta) + 1;
                neighbor = grid{ip,it};
                if ~isempty(neighbor)
                    if acos(dot(q,neighbor)) < r
                        ok = false;
                        break;
                    end
                end
            end
            if ~ok, break; end
        end

        if ok
            points(end+1,:) = q;
            activeList(end+1,:) = q;
            grid{phi,theta} = q;
            found = true;
            break;
        end
    end

    if ~found
        activeList(idx,:) = [];
    end
end
end
