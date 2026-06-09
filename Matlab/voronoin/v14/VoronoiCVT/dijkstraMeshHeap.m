function V = dijkstraMeshHeap(V, sites)
    % Dijkstra with a binary min-heap priority queue (lazy decrease-key).
    %
    % Complexity: O((E + K) log K) where K is number of heap pushes.
    % "Lazy" means we can have duplicates in heap; we skip stale entries.

    heap = BinaryMinHeap();

    for source = sites
        V(source).dist = 0;                 % initial point dist is 0
        V(source).origin_site = source;     % store the starting point
        heap.push(V(source).dist, source);
    end
    
    while ~heap.isEmpty()
        [current_index_dist, current_index] = heap.popMin();

        % Skip stale heap entries (we already found a better path)
        if current_index_dist > V(current_index).dist
            continue;
        end

        
        % Relax neighbors
        neigh_b = V(current_index).neigh;
        neigh_dist  = V(current_index).neighW;
        for j = 1:numel(neigh_b)
            v_neight = neigh_b(j);
           
            alt = current_index_dist + neigh_dist(j);
            if alt < V(v_neight).dist
                V(v_neight).dist = alt;
                V(v_neight).prev = current_index;
                V(v_neight).origin_site = V(current_index).origin_site;
                heap.push(alt, v_neight); % lazy decrease-key
            end
        end
    end
end