function [X, E] = thomson11(varargin)
% THOMSON11  Find ~minimum-energy arrangement of 11 points on the unit sphere
%
%  [X, E] = THOMSON11() runs the solver and plots the result. X is 11x3
%  (row i = Cartesian coordinates of point i). E is the final Coulomb energy.
%
% Optional name-value pairs:
%   'MaxIter'    - maximum iterations (default 20000)
%   'Tol'        - convergence tolerance on max displacement (default 1e-6)
%   'Seed'       - rng seed for reproducible starts (default 1)
%   'PlotEvery'  - plot intermediate steps every N iterations (default 0: no intermediate)
%
% Example:
%   [X,E] = thomson11('MaxIter',10000,'Seed',42);

% Parse inputs
p = inputParser;
addParameter(p,'MaxIter',20000,@(x)isnumeric(x)&&x>0);
addParameter(p,'Tol',1e-6,@(x)isnumeric(x)&&x>0);
addParameter(p,'Seed',1,@(x)isnumeric(x));
addParameter(p,'PlotEvery',0,@(x)isnumeric(x)&&x>=0);
addParameter(p,'nPoints',11,@(x)isnumeric(x)&&x>=0);
addParameter(p, 'Verbose',0, @(x)isnumeric(x));

parse(p,varargin{:});
MaxIter = p.Results.MaxIter;
Tol = p.Results.Tol;
n = p.Results.nPoints;
rng(p.Results.Seed);
verbose = p.Results.Verbose;

alpha0 = 0.1;          % initial step length (controls speed)
alphaDecay = 1e-4;     % decay per iter (so alpha = alpha0/(1+alphaDecay*iter))
jitterProb = 0.005;    % probability to add small random jitter to escape local minima
jitterScale = 1e-3;    % jitter magnitude
minAlpha = 1e-8;

% Initialize points: random on sphere (rows of X)
X = randn(n,3);
X = X ./ vecnorm(X,2,2);  % normalize each row

% Precompute pairs for vectorized distance calculations
[idx_i, idx_j] = ndgrid(1:n,1:n);
mask = idx_i ~= idx_j;

% Helper: compute energy
energy = @(Y) sum(1./sqrt(sum((Y(idx_i(mask),:)-Y(idx_j(mask),:)).^2,2)))/2;
% (divide by 2 because we double-count every pair in mask traversal)

% Main relaxation loop
prev_max_move = Inf;
for iter = 1:MaxIter
    % compute pairwise vectors and distances
    diff = permute(X,[1,3,2]) - permute(X,[3,1,2]); % n x n x 3, diff(i,j,:) = X(i,:)-X(j,:)
    dist = sqrt(sum(diff.^2,3));                    % n x n
    % Avoid divide-by-zero on diagonal by setting to Inf
    dist(1:n+1:end) = Inf;
    
    % Coulomb force magnitude on i from j: f = (1/r^2)
    % Force vector on i: sum_j (diff(i,j,:) ./ r^3)
    invr3 = 1./(dist.^3);                             % n x n
    invr3(1:n+1:end) = 0;
    % compute total force
    F = squeeze(sum(diff .* invr3(:,:,ones(1,3)),2));  % n x 3
    
    % Project forces to tangent plane at each point: F_t = F - (F . x) x
    xdotF = sum(X .* F,2);
    Ft = F - (xdotF(:,ones(1,3))).*X;
    
    % adaptive step size
    alpha = max(alpha0/(1 + alphaDecay*iter), minAlpha);
    
    % Move along tangent direction
    Xnew = X + alpha * Ft;
    % Re-normalize to sphere
    Xnew = Xnew ./ vecnorm(Xnew,2,2);
    
    % occasionally add tiny jitter to escape shallow local minima
    if rand < jitterProb
        J = jitterScale * randn(n,3);
        Xnew = Xnew + J;
        Xnew = Xnew ./ vecnorm(Xnew,2,2);
    end
    
    max_move = max(vecnorm(Xnew - X, 2, 2));
    X = Xnew;
    
    % optional intermediate plotting
    if p.Results.PlotEvery>0 && mod(iter,p.Results.PlotEvery)==0
        clf;
        plot_on_sphere(X,iter,energy(X));
        drawnow;
    end
    
    % convergence test
    if max_move < Tol
        break
    end
    prev_max_move = max_move;
end

E = energy(X);

if verbose == 0
    return;
end
% final plot

plot_on_sphere(X,iter,E);

fprintf('Finished after %d iterations, max_move = %.3e, energy = %.12f\n', iter, max_move, E);

end

function plot_on_sphere(X,iter,E)
figure('Color','w');
hold on;
R =1;
[XS, YS, ZS] = sphere(100);
surf(R*XS, R*YS, R*ZS, ...
    'FaceAlpha', 1.0, ...
    'EdgeColor', 'none', ...
    'FaceColor', [0.7647 0.7647 0.7647]);

% Plot latitude-longitude points
scatter3(X(:,1),X(:,2),X(:,3),15, 'k', 'filled');
% Formatting for paper
axis equal
axis off
set(gca, 'Visible', 'off')
set(gcf, 'Color', 'w');          % white background
set(gca, 'Position', [0 0 1 1]); % fill figure
view(0,90) 
exportgraphics(gcf, 'thomson_side.png', 'ContentType', 'vector');
hold off;
end
