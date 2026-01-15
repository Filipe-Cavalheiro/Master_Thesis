clearvars all; clc; close all

% Parameters
width = 40;
height = 40;
r = 1.01;  % Minimum distance between points

times = zeros(1,100);
start = clock;
% Generate points
while 1  
    points = poissonDiskSampling(width, height, r);
    disp(size(points))
    dim = size(points);
    if dim(1) == 1000
        break;
    end
end
etime(clock, start)

% Plot
figure;
scatter(points(:,1), points(:,2), 15, 'k', 'filled'); % black dots, size 15
axis equal;
xlim([0 width]);
ylim([0 height]);
