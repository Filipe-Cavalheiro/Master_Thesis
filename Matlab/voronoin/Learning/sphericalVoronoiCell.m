function vertices = sphericalVoronoiCell(agent_idx, agents, R)
% Robust spherical Voronoi cell for an agent on a sphere

p = agents(agent_idx,:);
nAgents = size(agents,1);

% Generate dense sphere points
[Xs,Ys,Zs] = sphere(30); % 30x30 resolution
V = [Xs(:), Ys(:), Zs(:)] * R;

% Only keep vertices in the agent's Voronoi cell
for j = 1:nAgents
    if j == agent_idx
        continue;
    end
    q = agents(j,:);
    % normal of great circle (half-plane boundary)
    n = cross(p, q);
    if norm(n) < 1e-12
        continue; % skip nearly coincident points
    end
    n = n / norm(n);
    
    % Ensure half-space points toward the agent
    if dot(cross(q,n), p) < 0
        n = -n;
    end
    
    keep = dot(V, repmat(n, size(V,1),1), 2) >= 0;
    V = V(keep,:);
end

if isempty(V)
    % fallback: just use the agent itself as vertex
    vertices = p;
else
    % remove duplicates and normalize to exact sphere
    vertices = unique(V,'rows');
    vertices = R * vertices ./ vecnorm(vertices,2,2);
end
end