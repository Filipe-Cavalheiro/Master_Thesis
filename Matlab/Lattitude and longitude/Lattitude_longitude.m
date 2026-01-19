clearvars; clc; close all;

%times = zeros(1,100);
% Generate points«
%for for_loop_of_time = 1:100
    %start = clock;
    % --- Parameters ---
    R = 5;          % Sphere radius
    k = 2;         % Number of intervals in latitude (adjust for resolution)
    delta = pi/k;   % Angular spacing in radians (180°/k)
    
    % --- Latitude and longitude ---
    theta = linspace(0, pi, k+1);   % Colatitude: 0 (north pole) to pi (south pole)
    phi_intervals = 2*k;            % Number of meridians at equator
    points = [];

    for i = 1:length(theta)
        if i == 1 || i == length(theta)
            % Poles: single point each
            x = 0;
            y = 0;
            z = R * cos(theta(i));
            points = [points; x, y, z];
        else
            % Number of points on this latitude (equal spacing in longitude)
            n_phi = round(phi_intervals * sin(theta(i))); % optional: denser near equator
            phi = linspace(0, 2*pi, n_phi+1);
            phi(end) = []; % remove duplicate at 2*pi
            for j = 1:length(phi)
                x = R * sin(theta(i)) * cos(phi(j));
                y = R * sin(theta(i)) * sin(phi(j));
                z = R * cos(theta(i));
                points = [points; x, y, z];
            end
        end
    end

%    times(for_loop_of_time) = etime(clock, start);
%end

% std(times) % 0.0055 sec
% figure;
% histogram(times);
%return
% --- Extract coordinates ---
x = points(:,1);
y = points(:,2);
z = points(:,3);

% --- Plot ---
figure;
hold on;

% Draw semi-transparent sphere
[XS, YS, ZS] = sphere(100);
surf(R*XS, R*YS, R*ZS, ...
    'FaceAlpha', 1.0, ...
    'EdgeColor', 'none', ...
    'FaceColor', [0.7647 0.7647 0.7647]);

% Plot latitude-longitude points
scatter3(x, y, z, 15, 'k', 'filled');

% Formatting for paper
axis equal
axis off
set(gca, 'Visible', 'off')
set(gcf, 'Color', 'w');          % white background
set(gca, 'Position', [0 0 1 1]); % fill figure
view(3) 
%exportgraphics(gcf, 'latlon_sphere_normal.png', 'ContentType', 'vector');
hold off;

disp(size(points))