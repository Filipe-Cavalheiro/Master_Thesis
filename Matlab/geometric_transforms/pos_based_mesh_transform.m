clc; clear; close all;

figure;
axis equal;
grid on;
hold on;

% Parameters
nTheta = 100;
nR = 50;
nFrames = 200;
r = 5;

% Grid (circle in YZ plane)
[theta, radius] = meshgrid(linspace(0, 2*pi, nTheta), linspace(0, r, nR));
x0 = zeros(size(theta));
y0 = radius .* cos(theta);
z0 = radius .* sin(theta);
surf_direction = [1, 0, 0];
sphere_center = [0,0,0];

% Sphere target
phi = pi - acos(radius / r)*2;
x_s = r * cos(phi);
y_s = r * sin(phi) .* cos(theta);
z_s = r * sin(phi) .* sin(theta);

% Initial surface
x_curr = x0;
y_curr = y0;
z_curr = z0;

h = surf(x_curr, y_curr, z_curr, 'FaceColor', 'interp', 'EdgeColor', 'none');

view(3);
camlight; lighting gouraud;

% Moving p
hp = plot3(-10, 0, 0, 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
p_x = linspace(-3, 10, nFrames);

for t = 1:nFrames
    p = [p_x(t), 0.3*cos(t), 0.3*sin(t)];
    
    dist_surface = dot(p, surf_direction);
    isOrigin = sphere_center(1) < 1e-9;
    
    if dist_surface > -(r/2) && dist_surface < 0 && isOrigin
        u_target = 0.5 - (-dist_surface / (r/2))/2;
    elseif dist_surface >= 0 && isOrigin
        u_target = 0.5 + (dist_surface / (r/2))/2;
    elseif dist_surface >= r && ~isOrigin
        u_target = 2;
    elseif dist_surface >= r/2 && ~isOrigin
        u_target = dist_surface / (r/2);
    else
        u_target = 0;
    end

    % --- split into two stages ---
    if u_target <= 1
        u1 = u_target;
        u2 = 0;
        sphere_center = [0, p(2), p(3)];
    else
        u1 = 1;
        u2 = u_target - 1;
        sphere_center = [p(1) - r/2 + u2*r/2, p(2), p(3)];
    end
    
    % --- Circle → cone ---
    x_cone = r * (1 - (radius/r).^2);
    y_cone = y0;
    z_cone = z0;
    
    x_tmp = (1-u1)*x0 + u1*x_cone;
    y_tmp = y_cone;
    z_tmp = z_cone;
    
    % --- Cone → sphere ---
    if u2 > 0
        x_tmp = (1-u2)*x_tmp + u2*x_s;
        y_tmp  = (1-u2)*y_tmp + u2*y_s;
        z_tmp  = (1-u2)*z_tmp + u2*z_s;
    end
    
    x_curr = x_tmp + sphere_center(1);
    y_curr = y_tmp + sphere_center(2);
    z_curr = z_tmp + sphere_center(3);

    % Update plot
    set(h, 'XData', x_curr, ...
           'YData', y_curr, ...
           'ZData', z_curr);

    set(hp, 'XData', p(1), 'YData', p(2), 'ZData', p(3));

    drawnow;
    pause(0.05);
end