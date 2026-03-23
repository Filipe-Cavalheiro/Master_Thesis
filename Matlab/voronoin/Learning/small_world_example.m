function small_world_consensus_demo()
    % SMALL_WORLD_CONSENSUS_DEMO - Demo of N agents with small-world connections
    % Agents move in 2D space, connections update based on small-world principle
    
    close all; clear; clc;
    
    % Parameters
    N = 30;                 % Number of agents
    k = 4;                  % Number of nearest neighbors (initial local connections)
    gamma = 0.2;            % Rewiring probability (0 = only local, 1 = fully random)
    speed = 0.02;           % Agent movement speed
    comm_range = 0.3;       % Maximum communication distance
    time_steps = 300;       % Simulation time steps
    update_interval = 5;    % Update connections every N steps
    
    % Initialize agents
    % Random initial positions in [0,1]x[0,1]
    positions = rand(N, 2);
    
    % Random initial values for consensus (e.g., temperature, opinion)
    values = rand(N, 1)*10;
    
    % Create figure
    figure('Position', [100, 100, 1200, 500]);
    
    for t = 1:time_steps
        % Update agent positions (random walk with boundary reflection)
        % Random movement
        movement = speed * (rand(N, 2) - 0.5);
        positions = positions + movement;
        
        % Reflect at boundaries [0,1]
        positions = max(0, min(1, positions));
        
        % Update small-world connections periodically
        if mod(t, update_interval) == 1 || t == 1
            connections = build_small_world_network(positions, k, gamma, comm_range);
        end
        
        % Run consensus step (only with connected neighbors)
        values = consensus_step(values, connections);
        
        % Visualization
        subplot(1, 2, 1);
        plot_network(positions, connections, values);
        title(sprintf('Time: %d | Small-World Network (γ=%.1f)', t, gamma));
        
        subplot(1, 2, 2);
        plot_consensus_progress(values, t);
        
        drawnow;
        
        % Small pause for animation
        pause(0.05);
    end
end

function connections = build_small_world_network(positions, k, gamma, comm_range)
    % BUILD_SMALL_WORLD_NETWORK - Creates connections using Watts-Strogatz model
    % Inputs:
    %   positions: Nx2 matrix of agent positions
    %   k: number of nearest neighbors
    %   gamma: rewiring probability
    %   comm_range: maximum communication distance
    % Output:
    %   connections: NxN logical adjacency matrix
    
    N = size(positions, 1);
    
    % Step 1: Create initial ring lattice (local connections)
    % Connect each agent to k nearest neighbors based on position
    dist_matrix = squareform(pdist(positions));
    
    % Start with no connections
    connections = false(N, N);
    
    % For each agent, find k nearest neighbors
    for i = 1:N
        [~, idx] = sort(dist_matrix(i, :));
        % Exclude self (first element) and take next k neighbors
        neighbors = idx(2:min(k+1, N));
        
        % Add connections (bidirectional)
        connections(i, neighbors) = true;
        connections(neighbors, i) = true;
    end
    
    % Step 2: Rewire connections with probability gamma
    for i = 1:N
        % Find current neighbors
        current_neighbors = find(connections(i, :));
        
        for j = 1:length(current_neighbors)
            neighbor = current_neighbors(j);
            
            % Only rewire if i < neighbor to avoid double processing
            if i < neighbor && rand() < gamma
                % Remove the connection
                connections(i, neighbor) = false;
                connections(neighbor, i) = false;
                
                % Find potential new connections (excluding self and current neighbors)
                available = setdiff(1:N, [i, current_neighbors]);
                
                if ~isempty(available)
                    % Pick random new neighbor
                    new_neighbor = available(randi(length(available)));
                    
                    % Check communication range constraint
                    if dist_matrix(i, new_neighbor) <= comm_range
                        % Add new connection
                        connections(i, new_neighbor) = true;
                        connections(new_neighbor, i) = true;
                    else
                        % If out of range, try to reconnect to someone in range
                        in_range = find(dist_matrix(i, :) <= comm_range & dist_matrix(i, :) > 0);
                        in_range = setdiff(in_range, [i, current_neighbors]);
                        
                        if ~isempty(in_range)
                            new_neighbor = in_range(randi(length(in_range)));
                            connections(i, new_neighbor) = true;
                            connections(new_neighbor, i) = true;
                        else
                            % No one in range, restore original connection
                            connections(i, neighbor) = true;
                            connections(neighbor, i) = true;
                        end
                    end
                end
            end
        end
    end
    
    % Ensure symmetry (just in case)
    connections = connections | connections';
end

function new_values = consensus_step(values, connections)
    % CONSENSUS_STEP - Each agent averages with connected neighbors
    N = length(values);
    new_values = zeros(N, 1);
    
    for i = 1:N
        % Find neighbors (including self)
        neighbors = [i, find(connections(i, :))];
        
        % Average values
        new_values(i) = mean(values(neighbors));
    end
    
    % Add small noise to prevent perfect convergence (for visualization)
    % new_values = new_values + 0.001 * randn(N, 1);
end

function plot_network(positions, connections, values)
    % PLOT_NETWORK - Visualize agents and their connections
    
    cla;
    hold on;
    
    N = size(positions, 1);
    
    % Plot connections
    for i = 1:N
        for j = i+1:N
            if connections(i, j)
                plot([positions(i,1), positions(j,1)], ...
                     [positions(i,2), positions(j,2)], ...
                     'b-', 'LineWidth', 0.5, 'Color', [0.7 0.7 1]);
            end
        end
    end
    
    % Plot agents
    scatter(positions(:,1), positions(:,2), 100, values, 'filled');
    colormap(jet);
    colorbar;
    caxis([0 10]);
    
    % Add agent numbers
    for i = 1:N
        text(positions(i,1)+0.01, positions(i,2)+0.01, num2str(i), ...
             'FontSize', 8, 'FontWeight', 'bold');
    end
    
    xlim([-0.1 1.1]);
    ylim([-0.1 1.1]);
    axis equal;
    grid on;
    xlabel('X Position');
    ylabel('Y Position');
    hold off;
end

function plot_consensus_progress(values, t)
    % PLOT_CONSENSUS_PROGRESS - Show how values converge over time
    
    persistent value_history;
    
    if t == 1
        value_history = [];
    end
    
    % Store current values
    value_history = [value_history; values'];
    
    % Plot last 100 steps
    start_idx = max(1, size(value_history, 1) - 100);
    plot(start_idx:size(value_history, 1), value_history(start_idx:end, :), 'LineWidth', 1);
    
    xlim([start_idx, size(value_history, 1) + 10]);
    xlabel('Time Step');
    ylabel('Agent Values');
    title('Consensus Progress');
    grid on;
    
    % Show current spread
    spread = std(values);
    text(size(value_history, 1), 0.9, sprintf('Std: %.3f', spread), ...
         'FontSize', 10, 'FontWeight', 'bold');
end