%% Computation Part
clc; clear; close all;

% Parameters
nPoints = 200;      % Number of points
nFrames = 200;      % Number of animation frames
r = 5;              % Base radius

% Generate circle points
theta = 2*pi*rand(nPoints,1);
radius = r * sqrt(rand(nPoints,1));
x = radius .* cos(theta);
y = radius .* sin(theta);
z = zeros(size(x));

% Precompute cone target heights
z_cone = zeros(nPoints, nFrames);
for t = 1:nFrames
    time_z = t/nFrames;
    z_cone(:,t) = r * time_z * (1 - sqrt(x.^2 + y.^2)/r);
end

% Precompute sphere coordinates
phi = pi - acos(radius / r)*2;        % polar angle for sphere
x_s = r * sin(phi) .* cos(theta);
y_s = r * sin(phi) .* sin(theta);
z_s = r * cos(phi);

% Precompute sphere animation frames
x_s_frames = zeros(nPoints, nFrames);
y_s_frames = zeros(nPoints, nFrames);
z_s_frames = zeros(nPoints, nFrames);
for t = 1:nFrames
    alpha = t / nFrames;
    x_s_frames(:,t) = (1-alpha)*x + alpha*x_s;
    y_s_frames(:,t) = (1-alpha)*y + alpha*y_s;
    z_s_frames(:,t) = (1-alpha)*z_cone(:,end) + alpha*z_s;
end

%% Animation Part
figure;
axis equal;
grid on;
hold on;

h = scatter3(x, y, z, 36, 'filled');
view(3);

% Animate circle → cone
for t = 1:nFrames
    set(h, 'ZData', z_cone(:,t));
    drawnow;
    pause(0.01);
end

pause(0.5);

% Animate cone → sphere
for t = 1:nFrames
    set(h, 'XData', x_s_frames(:,t), ...
           'YData', y_s_frames(:,t), ...
           'ZData', z_s_frames(:,t));
    drawnow;
    pause(0.01);
end