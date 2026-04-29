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