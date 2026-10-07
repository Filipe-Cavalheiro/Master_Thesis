clearvars all; clc; close all

% Parameters
k = 30;
r = 0.093;
verbose = false;

points = poissonDiskSampling(r, k);
sphere_uniformity(points)
if verbose == false
    return;
end

disp(length(points));

figure;
hold on;

% Draw semi-transparent sphere
[XS, YS, ZS] = sphere(100);   
surf(XS, YS, ZS, ...
    'FaceAlpha', 1.0, ...
    'EdgeColor', 'none', ...
    'FaceColor', [0.7647 0.7647 0.7647]); 

% Plot points
scatter3(points(:,1), points(:,2), points(:,3), 15, 'k', 'filled');  % small black filled points

% Formatting for paper
axis equal
axis off
set(gca, 'Visible', 'off')
set(gcf, 'Color', 'w');          % white background
set(gca, 'Position', [0 0 1 1]); % fill figure
view(90,0);
print2pdf("poisson_disk_side.pdf", 1)
view(0,90);
print2pdf("poisson_disk_top.pdf", 1)
view(3);
print2pdf("poisson_disk_normal.pdf", 1)
hold off;