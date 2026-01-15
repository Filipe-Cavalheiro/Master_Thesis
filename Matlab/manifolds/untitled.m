% Stereographic Projection of a Sphere onto a Plane
clear; clc; close all;

% Create a unit sphere
[u, v] = meshgrid(linspace(0, 2*pi, 100), linspace(0, pi, 50));
x = sin(v).*cos(u);
y = sin(v).*sin(u);
z = cos(v);

% Stereographic projection (from north pole onto plane z = -1)
x_proj = x ./ (1 - z);
y_proj = y ./ (1 - z);

% Plot 1: Original Sphere in 3D
figure('Name', '3D Sphere and Projection', 'NumberTitle', 'off')
subplot(1,2,1)
surf(x, y, z, 'FaceAlpha', 0.8, 'EdgeColor', 'none')
axis equal
xlabel('X'), ylabel('Y'), zlabel('Z')
title('Original Sphere (3D)')
colormap turbo
light, lighting gouraud
view(3)

% Plot 2: Projected 2D Plane
subplot(1,2,2)
surf(x_proj, y_proj, zeros(size(x_proj)), z, ...
    'FaceAlpha', 0.9, 'EdgeColor', 'none');
axis equal
xlabel('X'), ylabel('Y')
title('Stereographic Projection (2D Plane)')
colormap turbo
view(2)
colorbar
