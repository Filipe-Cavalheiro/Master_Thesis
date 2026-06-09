function edges = extractVoronoiEdges(V)
    edges = [];

    N = numel(V);

    for i = 1:N
        neigh_i = V(i).neigh;

        for k = 1:numel(neigh_i)
            j = neigh_i(k);

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