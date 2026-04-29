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