clearvars; close all; clc;

% Your data (100 samples each)
t1 = load("thomson_execution_times_per_iter.mat");
t2 = load("llyod_execution_times_per_iter.mat");
t3 = load("voronoi_half_plane_iterations_time.mat");

t1 = load("thomson_execution_time.mat");
t2 = load("voronoi_convex_hull_avg_time.mat");
t3 = load("voronoi_half_plane_time.mat");

% Combine data
data   = [t1.times'; t2.avg_time'; t3.times'];

groups = [ ...
    repmat("thomson",100,1);
    repmat("Llyod (convex hull)",100,1);
    repmat("Llyod (half plane)",100,1)];

% Plot
figure;
boxplot(data, groups);
xlabel('Analysis Method');
ylabel('Execution Time [Sec]');
title('Execution Time Distribution per Method');
grid on;

set(gcf,'Units','centimeters');
screenposition = get(gcf,'Position');

set(gcf, ...
    'PaperPosition',[0 0 screenposition(3:4)], ...
    'PaperSize',[screenposition(3:4)]);

print -dpdf -painters result_fig
