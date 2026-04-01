clearvars; close all; clc;

% Your data (100 samples each)
t1 = load("latlon_execution_times.mat");
t2 = load("fibonacci_execution_times.mat");
t3 = load("poisson_disk_execution_times.mat");

% Combine data
data   = [t1.times'; t2.times'; t3.times'];

groups = [ ...
    repmat("latlon",100,1);
    repmat("fibonacci",100,1);
    repmat("poisson",100,1);
    ];

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
