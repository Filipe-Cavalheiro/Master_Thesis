clear; clc; %close all;
% times = zeros(1,100);
% % Generate points
% for j = 1:100
%     start = clock;
    % Number of points
    P = 1001;                  
    N = floor((P-1)/2);         % Since P = 2N + 1 works best for odd P
    
    % Sphere radius
    R = 5;
    
    % Golden ratio
    phi_golden = (1 + sqrt(5)) / 2;
    
    % Generate indices
    idx = -N:N;
    
    % Latitude and phi
    theta_pos = acos(2*idx/P);      % latitude (eq: 2.4, da tese)

    theta = diff(theta_pos);
    
    phi_pos = 2*pi*idx/phi_golden;  % longitude (eq: 2.5, da tese)
    if size(phi_pos) < 2
        phi = 0;
    else
        phi = diff(phi_pos);
        phi = phi(1) ;
    end
    
    theta_tmp = theta_pos(1);
    phi_tmp = phi_pos(1);
    
    x = zeros(1,P-1);
    y = zeros(1,P-1);
    z = zeros(1,P-1);
    
    for i = 1:P-1
        % Cartesian coordinates of Fibonacci points
        x(i) = R * sin(theta_tmp) .* cos(phi_tmp);
        y(i) = R * sin(theta_tmp) .* sin(phi_tmp);
        z(i) = R * cos(theta_tmp);
        theta_tmp = theta_tmp + theta(i);
        phi_tmp = phi_tmp + phi;
    end
    x(P) = R * sin(theta_tmp) .* cos(phi_tmp);
    y(P) = R * sin(theta_tmp) .* sin(phi_tmp);
    z(P) = R * cos(theta_tmp);
    
    points = [x(:), y(:), z(:)];

%     times(j) = etime(clock, start);
% end
% 
% std(times) % 6.6848e-04 sec
% figure;
% histogram(times);
% % --- Plot ---
figure;
hold on;

% Draw semi-transparent sphere
[XS, YS, ZS] = sphere(100);   
surf(R*XS, R*YS, R*ZS, ...
    'FaceAlpha', 1.0, ...
    'EdgeColor', 'none', ...
    'FaceColor', [0.7647 0.7647 0.7647]); 

% Plot points
scatter3(x, y, z, 15, 'k', 'filled');

% Formatting for paper
axis equal
axis off
set(gca, 'Visible', 'off')
set(gcf, 'Color', 'w');          % white background
set(gca, 'Position', [0 0 1 1]); % fill figure
view(3);

%exportgraphics(gcf, 'fibonacci_sphere_top.png', 'ContentType', 'vector');
hold off;
