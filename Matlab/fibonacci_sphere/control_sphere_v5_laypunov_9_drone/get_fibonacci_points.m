function fibonacci_points = get_fibonacci_points(P, R)
    % This function returns the desired trajectory for the drones to form a 
    % sphere around a point
    % input parameters:
    % P - number of drones
    % R - radios of sphere
    if P < 3
        error("Not enought points")
    end

    % Number of fibonacci_points           
    N = floor((P-1)/2);     % Since P = 2N + 1 works best for odd P
    K = 3;                  % nearest neighbor
    
    % Golden ratio
    phi_golden = (1 + sqrt(5)) / 2;
    
    % Generate indices
    idx = -N:N;
    
    % Latitude and phi
    theta_pos = acos(2*idx/P);      % latitude
    theta = diff(theta_pos);
    phi_pos = 2*pi*idx/phi_golden;  % longitude
    phi = diff(phi_pos);
    phi = phi(1) ;
    
    theta_tmp = theta_pos(1);
    phi_tmp = phi_pos(1);
    
    x_fibonacci = zeros(1,P-1);
    y_fibonacci = zeros(1,P-1);
    z_fibonacci = zeros(1,P-1);
    
    for i = 1:P-1
        % Cartesian coordinates of Fibonacci fibonacci_points
        x_fibonacci(i) = R * sin(theta_tmp) .* cos(phi_tmp);
        y_fibonacci(i) = R * sin(theta_tmp) .* sin(phi_tmp);
        z_fibonacci(i) = R * cos(theta_tmp);
        theta_tmp = theta_tmp + theta(i);
        phi_tmp = phi_tmp + phi;
    end
    x_fibonacci(P) = R * sin(theta_tmp) .* cos(phi_tmp);
    y_fibonacci(P) = R * sin(theta_tmp) .* sin(phi_tmp);
    z_fibonacci(P) = R * cos(theta_tmp);
    
    fibonacci_points = [x_fibonacci(:), y_fibonacci(:), z_fibonacci(:)];
end