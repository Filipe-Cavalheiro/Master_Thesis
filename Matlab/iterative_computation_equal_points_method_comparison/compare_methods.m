clearvars; close all; clc;

t1 = load("thomson_time.mat");
t2 = load("voronoi_conv_hull_time.mat");
t3 = load("voronoi_half_plane_time.mat");

% data   = [t1.thomson_times'; t2.convex_hull_times'; t3.half_plane_times'];
% 
% groups = [ ...
%     repmat("thomson",100,1);
%     repmat("Llyod (convex hull)",100,1);
%     repmat("Llyod (half plane)",100,1)];

data   = [t1.thomson_times'; t2.convex_hull_times'];

groups = [ ...
    repmat("thomson",100,1);
    repmat("Llyod (convex hull)",100,1)];

% Plot
figure;
boxplot(data, groups);
xlabel('Analysis Method');
ylabel('Execution Time [Sec]');
grid on;

set(gcf,'Units','centimeters');
screenposition = get(gcf,'Position');

set(gcf, ...
    'PaperPosition',[0 0 screenposition(3:4)], ...
    'PaperSize',[screenposition(3:4)]);

print2pdf("iterative_run_comparing_methods_box_shape_2.pdf", true)
