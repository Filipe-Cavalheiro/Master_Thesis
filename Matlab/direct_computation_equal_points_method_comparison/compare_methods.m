clearvars; close all; clc;

% Load data
t2 = load("fibonacci_execution_times.mat");

% Data
data = t2.times';

% Create figure
figure;

% -----------------------------
% LEFT: Boxplot WITH outliers
% -----------------------------
subplot(1,2,1)

boxplot(data, 'Symbol','r+'); % show outliers
set(gca,'XTickLabel',{})    % remove "1"
xlabel('With Outliers');


ylabel('Execution Time [ms]');
grid on;

% -----------------------------
% RIGHT: Boxplot WITHOUT outliers
% -----------------------------
subplot(1,2,2)

boxplot(data, 'Symbol',''); % hide outliers
set(gca,'XTickLabel',{})    % remove "1"
xlabel('Without Outliers');
grid on;


% Compute limits excluding outliers
q1 = prctile(data,25);
q3 = prctile(data,75);
iqr_val = q3 - q1;

lower_lim = q1 - 1.5*iqr_val;
upper_lim = q3 + 1.5*iqr_val;

ylim([lower_lim upper_lim])

% Global title
sgtitle('Execution Time Distribution');

% Figure export settings
set(gcf,'Units','centimeters');
screenposition = get(gcf,'Position');

set(gcf, ...
    'PaperPosition',[0 0 screenposition(3:4)], ...
    'PaperSize',[screenposition(3:4)]);

print -dpdf -painters fibonacci_exec_time