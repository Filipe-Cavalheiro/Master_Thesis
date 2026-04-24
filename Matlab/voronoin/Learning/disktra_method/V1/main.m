clear; clc; close all;

% Parameters
state.mesh_resolution = 40;
state.numSites = 8;
state.maxIter = 20;

state.obj_index = 1;
state.objList = {'circle','sphere','ellipsoid','paraboloid','frog.stl','bust.stl'};
state.vertices = [];
state.faces = [];
state.sites = []; 


disp('Controls:');
disp('↑  : new random sites (same object)');
disp('←/→: switch object');
disp('Enter: run CVT');
            
[state.vertices, state.faces] = loadObject(state);
N = size(state.vertices,1);
state.sites = randperm(N, state.numSites);

hFig = figure('KeyPressFcn', @keyHandler);
state.hFig = hFig;
hold on; view(3);

gfx.surf = trisurf(state.faces, state.vertices(:,1), state.vertices(:,2), state.vertices(:,3), ...
    zeros(size(state.vertices,1),1), ...
    'EdgeColor','none');

axis equal;
camlight;
lighting gouraud;
shading interp
lightangle(-45,30)
h.AmbientStrength = 0.3;
h.DiffuseStrength = 0.8;
h.SpecularStrength = 0.9;
h.SpecularExponent = 25;

gfx.sites = scatter3(nan, nan, nan, 60, 'g', 'x');
gfx.voronoi_edges = scatter3(nan, nan, nan, 60, 'r', 'x');

data.state = state;
data.gfx   = gfx;
guidata(hFig, data);
visualizeState(state, 'New random sites', gfx);

%% --- functions
function V = buildVertexObjects(vertices, faces)
% Build MeshVertex array + adjacency from triangle faces.

    N = size(vertices,1);
    V(N,1) = MeshVertex; % preallocate object array
    for i = 1:N
        V(i) = MeshVertex(i, vertices(i,:));
    end

    % Unique undirected edges from faces
    E = [faces(:,[1 2]); faces(:,[2 3]); faces(:,[3 1])];
    E = sort(E, 2);
    E = unique(E, 'rows');

    % Fill adjacency lists with weights (edge lengths)
    for k = 1:size(E,1)
        a = E(k,1); 
        b = E(k,2);
        w = norm(vertices(a,:) - vertices(b,:), 2);

        V(a).neigh(end+1,1)  = b;
        V(a).neighW(end+1,1) = w;

        V(b).neigh(end+1,1)  = a;
        V(b).neighW(end+1,1) = w;
    end
end

function V = dijkstraMeshHeap(V, sites)
    % Dijkstra with a binary min-heap priority queue (lazy decrease-key).
    %
    % Complexity: O((E + K) log K) where K is number of heap pushes.
    % "Lazy" means we can have duplicates in heap; we skip stale entries.

    heap = BinaryMinHeap();

    for source = sites
        V(source).dist = 0;
        V(source).origin_site = source;
        heap.push(V(source).dist, source);
    end
    
    while ~heap.isEmpty()
        [current_index_dist, current_index] = heap.popMin();

        % Skip stale heap entries (we already found a better path)
        if current_index_dist > V(current_index).dist
            continue;
        end

        
        % Relax neighbors
        nb = V(current_index).neigh;
        w  = V(current_index).neighW;
        for j = 1:numel(nb)
            v = nb(j);
           
            alt = current_index_dist + w(j);
            if alt < V(v).dist
                V(v).dist = alt;
                V(v).prev = current_index;
                V(v).origin_site = V(current_index).origin_site;
                heap.push(alt, v); % lazy decrease-key
            end
        end
    end
end

function sites = updateSitesCVT(vertices, faces, V, sites)
    
    orig = [V.origin_site]';
    numSites = numel(sites);
    
    centroids = zeros(numSites,3);
    areas     = zeros(numSites,1);
    
    for f = 1:size(faces,1)
        tri = faces(f,:);
        s = orig(tri);
    
        % only take triangles fully inside one region
        if s(1)==s(2) && s(2)==s(3)
            k = find(sites == s(1));
            if isempty(k), continue; end
    
            p1 = vertices(tri(1),:);
            p2 = vertices(tri(2),:);
            p3 = vertices(tri(3),:);
    
            % triangle centroid
            c = (p1 + p2 + p3)/3;
    
            % triangle area
            A = 0.5 * norm(cross(p2-p1, p3-p1));
    
            centroids(k,:) = centroids(k,:) + A * c;
            areas(k)       = areas(k) + A;
        end
    end
    
    % finalize centroids
    newSites = zeros(size(sites));
    
    for k = 1:numSites
        if areas(k) == 0
            newSites(k) = sites(k);
            continue;
        end
    
        c = centroids(k,:) / areas(k);
    
        % snap to closest vertex
        d = vecnorm(vertices - c, 2, 2);
        [~, newSites(k)] = min(d);
    end
    
    sites = unique(newSites);
end

function [vertices, faces] = generateMesh(shapeType, res)

    switch lower(shapeType)

        case 'circle'
            [theta, radius] = meshgrid(linspace(0, 2*pi, res), linspace(0, 1, res));
            x = radius .* cos(theta);
            y = radius .* sin(theta);
            z = zeros(size(x));

            [faces, vertices] = surf2patch(x, y, z, 'triangles');

        case 'sphere'
            [X,Y,Z] = sphere(res);
            [faces, vertices] = surf2patch(X,Y,Z,'triangles');

            % normalize (important if using sphere())
            vertices = vertices ./ vecnorm(vertices,2,2);

        case 'ellipsoid'
            a = 1; b = 0.6; c = 0.4; % axis scales

            [X,Y,Z] = sphere(res);
            X = a * X;
            Y = b * Y;
            Z = c * Z;

            [faces, vertices] = surf2patch(X,Y,Z,'triangles');

        case 'paraboloid'
            [r, theta] = meshgrid(linspace(0,1,res), linspace(0,2*pi,res));

            x = r .* cos(theta);
            y = r .* sin(theta);
            z = r.^2;   % paraboloid z = r^2

            [faces, vertices] = surf2patch(x,y,z,'triangles');

        otherwise
            error('Unknown shape type');
    end

    % --- Weld duplicates (important for all shapes)
    tol = 1e-10;
    [verticesW, ~, ic] = uniquetol(vertices, tol, 'ByRows', true);

    facesW = ic(faces);

    % Remove degenerate faces
    deg = facesW(:,1)==facesW(:,2) | facesW(:,2)==facesW(:,3) | facesW(:,1)==facesW(:,3);
    facesW(deg,:) = [];

    vertices = verticesW;
    faces    = facesW;
end

function runCVT(state, gfx)
    vertices = state.vertices;
    faces    = state.faces;
    sites    = state.sites;    
    for iter = 1:state.maxIter
        V = buildVertexObjects(vertices, faces);
        V = dijkstraMeshHeap(V, sites);

        % visualization
        N = length(V);
        dist = zeros(N,1);
        for i = 1:N
            dist(i) = V(i).dist;
        end

        set(gfx.surf, 'CData', dist);
 
        pts = vertices(sites,:);
        set(gfx.sites, ...
            'XData', pts(:,1), ...
            'YData', pts(:,2), ...
            'ZData', pts(:,3));

        edges = extractVoronoiEdges(V);

        if ~isempty(edges)
            P1 = vertices(edges(:,1),:);
            P2 = vertices(edges(:,2),:);
        
            X = [P1(:,1) P2(:,1)]';
            Y = [P1(:,2) P2(:,2)]';
            Z = [P1(:,3) P2(:,3)]';
        
            set(gfx.voronoi_edges, ...
                'XData', X(:), ...
                'YData', Y(:), ...
                'ZData', Z(:));
        end

        title(sprintf('Iteration %d', iter));
        drawnow;
        
        
        newSites = updateSitesCVT(vertices, faces, V, sites);

        if isequal(sort(newSites), sort(sites))
            fprintf('Converged at iteration %d\n', iter);
            break;
        end
        sites = newSites;
    end
end

function visualizeState(state, ttl, gfx)

    V = state.vertices;
    F = state.faces;
    N = size(V,1);

    set(gfx.surf, ...
        'Faces', F, ...
        'Vertices', V, ...
        'FaceVertexCData', zeros(N,1));

    pts = V(state.sites,:);

    set(gfx.voronoi_edges, ...
        'XData', nan, ...
        'YData', nan, ...
        'ZData', nan);

    set(gfx.sites, ...
        'XData', pts(:,1), ...
        'YData', pts(:,2), ...
        'ZData', pts(:,3));

    title(sprintf('%s | Object: %s', ttl, state.objList{state.obj_index}));
end

function [vertices, faces] = loadObject(state)

    name = state.objList{state.obj_index};

    if state.obj_index < 5
        [vertices, faces] = generateMesh(name, state.mesh_resolution);
    else
        [F, ~] = stlread(name);
        vertices = F.Points;
        faces = F.ConnectivityList;
    end
end


function keyHandler(src, event)
    data = guidata(src);
    state = data.state;
    gfx   = data.gfx;

    switch event.Key
        case 'uparrow'
            % same object, new random sites
            [state.vertices, state.faces] = loadObject(state);
            N = size(state.vertices,1);
            state.sites = randperm(N, state.numSites);
            visualizeState(state, 'New random sites', gfx);

        case 'rightarrow'
            % next object
            state.obj_index = state.obj_index + 1;
            if state.obj_index > numel(state.objList)
                state.obj_index = 1;
            end
            [state.vertices, state.faces] = loadObject(state);
            state.sites = randperm(size(state.vertices,1), state.numSites);
            visualizeState(state, 'Next object', gfx);

        case 'leftarrow'
            % previous object
            state.obj_index = state.obj_index - 1;
            if state.obj_index < 1
                state.obj_index = numel(state.objList);
            end
            [state.vertices, state.faces] = loadObject(state);
            state.sites = randperm(size(state.vertices,1), state.numSites);
            visualizeState(state, 'Previous object', gfx);

        case 'return'
            % run algorithm
            runCVT(state, gfx);

        otherwise
            return;
    end

    data.state = state;
    guidata(src, data);
end

function edges = extractVoronoiEdges(V)
    edges = [];

    N = numel(V);

    for i = 1:N
        ni = V(i).neigh;

        for k = 1:numel(ni)
            j = ni(k);

            % avoid duplicate edges
            if j <= i
                continue;
            end

            if V(i).origin_site ~= V(j).origin_site
                edges(end+1, :) = [i j]; %#ok<AGROW>
            end
        end
    end
end