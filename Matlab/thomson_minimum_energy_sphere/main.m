clearvars all; clc; close all


% Generate points
% for i = 1:100
%     start = clock;
    [X,E, iter] = thomson11('MaxIter',100,'Seed',0,'PlotEvery',0, 'nPoints', 30, 'Verbose', 1);
%     times(i) = etime(clock, start);
% end