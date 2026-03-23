function [connections, Laplacian, degree_matrix] = build_small_world_network_geo(positions, k_neighbours, gamma, comm_range, sphere_center, Param)
    % BUILD_SMALL_WORLD_NETWORK - Creates connections using Watts-Strogatz model
    % Returns adjacency matrix, Laplacian matrix, and degree matrix 
    
    % Step 1: Create adjacency matrix (connections)
    % Start with no connections
    A = false(Param.nD, Param.nD); 
    
    % For each agent, find k nearest neighbors
    dist_matrix = geo_distances(positions, Param.sphere_radius, sphere_center);
    for i = 1:Param.nD
        [~, idx] = sort(dist_matrix(i, :));
        idx(end) = []; 
        neighbors = idx(1:min(k_neighbours, Param.nD-1));
        A(i, neighbors) = true;
    end
    
    % Step 2: Rewire connections with probability gamma
    for i = 1:Param.nD
        current_neighbors = find(A(i, :));

        for j = 1:length(current_neighbors)
            neighbor = current_neighbors(j);

            if i < neighbor && rand() < gamma
                A(i, neighbor) = false;

                available = setdiff(1:Param.nD, [i, current_neighbors]);

                if ~isempty(available)
                    new_neighbor = available(randi(length(available)));

                    if dist_matrix(i, new_neighbor) <= comm_range
                        A(i, new_neighbor) = true;
                    else
                        in_range = find(dist_matrix(i, :) <= comm_range & dist_matrix(i, :) > 0);
                        in_range = setdiff(in_range, [i, current_neighbors]);

                        if ~isempty(in_range)
                            new_neighbor = in_range(randi(length(in_range)));
                            A(i, new_neighbor) = true;
                        else
                            A(i, neighbor) = true;
                        end
                    end
                end
            end
        end
    end
    
    A = A|A';

    % Step 3: Compute degree matrix and Laplacian
    % Degree matrix D (diagonal)
    degrees = sum(A, 2);
    degree_matrix = diag(degrees);
    
    % Laplacian matrix L = D - A
    % Convert A to double for Laplacian calculation
    A_double = double(A);
    Laplacian = degree_matrix - A_double;
    
    % Return adjacency matrix as connections (for backward compatibility)
    connections = A;
end