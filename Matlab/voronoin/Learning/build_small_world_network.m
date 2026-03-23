function [connections, Laplacian, degree_matrix] = build_small_world_network(positions, k_neighbours, gamma, comm_range)
    % BUILD_SMALL_WORLD_NETWORK - Creates connections using Watts-Strogatz model
    % Returns adjacency matrix, Laplacian matrix, and degree matrix
    
    % Handle different input formats
    if ndims(positions) == 3
        positions = squeeze(positions);
    end
    
    N_agents = size(positions, 1);
    
    % Step 1: Create adjacency matrix (connections)
    % Start with no connections
    A = false(N_agents, N_agents); 
    
    % For each agent, find k nearest neighbors
    dist_matrix = squareform(pdist(positions));
    for i = 1:N_agents
        [~, idx] = sort(dist_matrix(i, :));
        neighbors = idx(2:min(k_neighbours+1, N_agents));
        A(i, neighbors) = true;
    end
    
    % Step 2: Rewire connections with probability gamma
    for i = 1:N_agents
        current_neighbors = find(A(i, :));
        
        for j = 1:length(current_neighbors)
            neighbor = current_neighbors(j);
            
            if i < neighbor && rand() < gamma
                A(i, neighbor) = false;
                
                available = setdiff(1:N_agents, [i, current_neighbors]);
                
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
    
    % Ensure symmetry (undirected graph)
    A = A | A';
    
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