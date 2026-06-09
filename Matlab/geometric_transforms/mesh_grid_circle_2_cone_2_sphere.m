%% Computation Part
clc; clear; close all;

% Parameters
nTheta = 100;    % Angular resolution
nR = 50;         % Radial resolution
nFrames = 200;   % Number of animation frames
r = 5;           % Base radius

% Generate circle grid (polar coordinates)
[theta, radius] = meshgrid(linspace(0, 2*pi, nTheta), linspace(0, r, nR));
x = radius .* cos(theta);
y = radius .* sin(theta);
z = zeros(size(x));

% Precompute cone heights
z_cone = zeros([size(x), nFrames]);
for t = 1:nFrames
    time_z = t / nFrames;
    z_cone(:,:,t) = r * time_z * (1 - radius/r);
end

% Precompute sphere coordinates
phi = pi - acos(radius / r)*2;   % polar angle for sphere
x_s = r * sin(phi) .* cos(theta);
y_s = r * sin(phi) .* sin(theta);
z_s = r * cos(phi);

% Precompute sphere animation frames
x_s_frames = zeros([size(x), nFrames]);
y_s_frames = zeros([size(y), nFrames]);
z_s_frames = zeros([size(z), nFrames]);
for t = 1:nFrames
    alpha = t / nFrames;
    x_s_frames(:,:,t) = (1-alpha)*x + alpha*x_s;
    y_s_frames(:,:,t) = (1-alpha)*y + alpha*y_s;
    z_s_frames(:,:,t) = (1-alpha)*z_cone(:,:,end) + alpha*z_s;
end

%% Animation Part
figure;
axis equal;
grid on;
hold on;

h = surf(x, y, z, 'FaceColor', 'interp', 'EdgeColor', 'none');
view(3);
camlight; lighting gouraud;

% Animate circle → cone
for t = 1:nFrames
    set(h, 'ZData', z_cone(:,:,t));
    drawnow;
    pause(0.01);
end

pause(0.5);

% Animate cone → sphere
for t = 1:nFrames
    set(h, 'XData', x_s_frames(:,:,t), ...
           'YData', y_s_frames(:,:,t), ...
           'ZData', z_s_frames(:,:,t));
    drawnow;
    pause(0.01);
end