function new_values = consensus_step(values, Laplacian, dt)
    % CONSENSUS_STEP_LAPLACIAN - Consensus using Laplacian dynamics
    % dx/dt = -L x
    % Discretized: x_new = x - dt * L * x
    
    if nargin < 3
        dt = 0.1;  % Time step
    end
    
    % Continuous-time consensus
    new_values = values - dt * Laplacian * values;
    
    % Ensure values stay bounded (optional)
    new_values = max(0, min(1, new_values));
end