clear; clc; close all;

% Define world grid
world_X = linspace(-5,5,100);
world_Y = linspace(-5,5,100);
[X,Y] = meshgrid(world_X, world_Y);

% Saddle surface
Z = X.^2 - Y.^2;

figure
surf(X,Y,Z,'EdgeColor','none')
hold on

xlabel('world\_X');
ylabel('world\_Y');
zlabel('Z');
title('Saddle Surface');
view(3);
grid on;