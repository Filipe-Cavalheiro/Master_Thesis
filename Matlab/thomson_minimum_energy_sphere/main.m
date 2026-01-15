clearvars all; clc; close all

times = zeros(1,100);
% Generate points«
for i = 1:100
    start = clock;
    [X,E] = thomson11('MaxIter',20000,'Seed',i,'PlotEvery',0, 'nPoints', 31, 'Verbose', 0);
    times(i) = etime(clock, start);
end

std(times) %0.1130 sec
figure;
histogram(times);