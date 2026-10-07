clearvars all; clc; close all

Nruns = 100;
thomson_times      = zeros(1,Nruns);
convergence_iters  = nan(1,Nruns);     % NaN for runs that did not converge
converged_count    = 0;
not_converged_count= 0;

%Generate points
for run = 1:Nruns
    start = clock;
    [X,E, iter, converged] = thomson11('MaxIter',1000,'Seed',0,'PlotEvery',0, 'nPoints', 30, 'Tol',0.1, 'Verbose', 0);
    thomson_times(run) = etime(clock, start);
    % sphere_uniformity(X);

    fprintf("Run %03d/%03d done in %.3f s\n", run, Nruns, thomson_times(run));
    if converged
        converged_count = converged_count + 1;
        convergence_iters(run) = iter;
    else
        not_converged_count = not_converged_count + 1;
    end

end

fprintf("Mean time per run: %.3f s\n", mean(thomson_times));
fprintf("Mean necessary iterations for convergence: %.3f \n", mean(convergence_iters));
fprintf("converged %d times \n", converged_count)
fprintf("did not converged %d times \n", not_converged_count)