clearvars all; clc; close all

% Parameters
width = 40;
height = 40;
r = 1.0;  % Minimum distance between points

times = zeros(1,100);

for i = 1:length(times)
    % Generate points
    start = clock;
    [points,iterCount] = poissonDiskSampling(width, height, r);
    total_time = etime(clock, start);
    times(i) = total_time / iterCount;
end
% Plot
figure;
scatter(points(:,1), points(:,2), 15, 'k', 'filled'); % black dots, size 15
axis equal;
xlim([0 width]);
ylim([0 height]);
