function newSite = runCVT(state, sites, gfx)
    % NEWSITES updates the position of sites on the mesh based on disktra
    % and the new centroid based on the area of the Voronoi section
    % Input:
    % state - contains variables associated with the mesh
    % sites - current location of the sites
    % gfx - grafical update (only on gfx.verbose = true)
    % output:
    % newSite - site of the updated site 
    % !! Only the site in index 1 of the sites is updated!!
    
    vertices = state.vertices;
    faces    = state.faces;
    
    V = buildVertexObjects(vertices, faces);
    V = dijkstraMeshHeap(V, sites);
    
    if gfx.verbose == true
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

        drawnow;
    end

    newSite = updateSitesCVT(vertices, faces, V, sites);
end