clc; clear; close all;
% Parameters
nFrames = 200;
Param.sphere_radius = 5;
P = 1001;                  

% --- pre_compute circle  ---
N = floor((P-1)/2);             % Since P = 2N + 1 works best for odd P
golden_angle = pi * (3 - sqrt(5));
x0 = zeros(P,1);
y0 = zeros(P,1);
z0 = zeros(P,1);

for k = 1:P
    r = Param.sphere_radius * sqrt(k / N);        % radius scaling
    theta = k * golden_angle;   % golden angle increment
    
    y0(k) = r * cos(theta);
    z0(k) = r * sin(theta);
end
v_circle = [x0(:), y0(:), z0(:)];

% --- pre_compute cone  ---
coneMap = @(V) [ ...
    Param.sphere_radius * (1 - (sqrt(V(:,2).^2 + V(:,3).^2)/Param.sphere_radius).^2), ...
    V(:,2), ...
    V(:,3)];

v_cone = coneMap(v_circle);

% --- pre_compute sphere  ---
phi_golden = (1 + sqrt(5)) / 2; % Golden ratio

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

i = (0:P-1)';
phi = 2*pi*i/phi_golden;
theta = acos(1 - 2*(i+0.5)/P);

x_s = Param.sphere_radius * cos(theta);
y_s = Param.sphere_radius * sin(theta).*sin(phi);
z_s = Param.sphere_radius * sin(theta).*cos(phi);
v_sphere = [x_s(:), y_s(:), z_s(:)];

surf_direction = [1, 0, 0];
sphere_center = [0,0,0];

% Initial surface
x_curr = x0;
y_curr = y0;
z_curr = z0;

% Moving p
p_x = linspace(-3, 10, nFrames);
mesh_pos = [0;0;0];

figure;
axis equal;
grid on;
hold on;

h_mesh = patch('Faces', [], ...
               'Vertices', v_circle(2:3, :), ...
               'FaceColor', [0.2 0.6 1.0], ...
               'EdgeColor', [0.2 0.2 0.2], ...
               'FaceAlpha', 0.8);

h_dot = scatter3(p_x(1), 0.3*cos(1), 0.3*cos(1), 50, 'r', 'filled');

view(3);
camlight;
lighting gouraud;

for t = 1:nFrames
    p = [p_x(t), 0.3*cos(t), 0.3*sin(t)];
    
    % Distance along surface normal
    dist_surface = dot(p, surf_direction);
    isOrigin = mesh_pos(1) < 1e-9;

    % Compute u_target
    if dist_surface > -(Param.sphere_radius/2) && dist_surface < 0 && isOrigin
        u_target = 0.5 - (-dist_surface / (Param.sphere_radius/2))/2;
    elseif dist_surface >= 0 && isOrigin
        u_target = 0.5 + (dist_surface / (Param.sphere_radius/2))/2;
    elseif dist_surface >= Param.sphere_radius && ~isOrigin
        u_target = 2;
    elseif dist_surface >= Param.sphere_radius/2 && ~isOrigin
        u_target = dist_surface / (Param.sphere_radius/2);
    else
        u_target = 0;
    end

    % Split stages
    if u_target <= 1
        u1 = u_target;
        u2 = 0;
        mesh_pos = [0; p(2); p(3)];
    else
        u1 = 1;
        u2 = u_target - 1;
        mesh_pos = [p(1) - Param.sphere_radius/2 + u2*Param.sphere_radius/2; p(2); p(3)];
    end

    % --- Circle → cone (operate per vertex) ---
    v_current = (1-u1).*v_circle + u1.*v_cone;
    
    if u2 > 0
        v_current = (1-u2).*v_current + u2.*v_sphere;
    end
    
    vertices = v_current + mesh_pos';

    if u2 >= 0.5
        faces = convhull(vertices(:,1), vertices(:,2), vertices(:,3));
    else
        DT = delaunayTriangulation(v_circle(:,2), v_circle(:,3));
        faces = DT.ConnectivityList;
    end

    % Update plot
    set(h_mesh, 'Faces', faces, 'Vertices', vertices);
    set(h_dot, 'XData', p(1), 'YData', p(2), 'ZData', p(3));

    drawnow;
    pause(0.05);
end