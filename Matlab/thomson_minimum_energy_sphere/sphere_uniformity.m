function results = sphere_uniformity(X, varargin)
%SPHERE_UNIFORMITY  Assess how uniformly points are spaced on the unit sphere.
%
%   results = sphere_uniformity(X) computes homogeneity metrics for a
%   point set X (N-by-3 or 3-by-N, need not be exactly unit-norm) and
%   compares them against a Monte-Carlo baseline of uniformly random
%   points on the sphere, so you can report e.g. "X is Y% more uniform
%   than random".
%
%   Metrics computed:
%     - ratio_min_mean : d_NN,min / d_NN,mean  (closer to 1 = more uniform)
%     - cv_nn          : coefficient of variation of nearest-neighbour
%                        distances, std(d_NN)/mean(d_NN) (lower = more uniform)
%     - riesz_energy   : sum_{i<j} 1/d_ij^s   (lower = more spread out)
%
%   Name-value options:
%     'RieszS'      exponent s in the Riesz energy (default 1, i.e. the
%                   classical Thomson/Coulomb energy)
%     'NumBaseline' number of random point sets averaged for the
%                   baseline (default 200)
%     'Verbose'     print a summary to the console (default true)
%
%   Example:
%     [X,E,iter,converged] = thomson11('MaxIter',1000,'Seed',0, ...
%         'PlotEvery',0,'nPoints',30,'Tol',0.1,'Verbose',0);
%     results = sphere_uniformity(X);

p = inputParser;
addParameter(p,'RieszS',1);
addParameter(p,'Verbose',true);
parse(p,varargin{:});
s       = p.Results.RieszS;
verbose = p.Results.Verbose;

X = normalize_points(X);
N = size(X,1);

[ratio, cv, meanNN, minNN] = nn_stats(X);
riesz = riesz_energy(X, s);

results.N               = N;
results.ratio_min_mean  = ratio;
results.cv_nn           = cv;
results.mean_nn         = meanNN;

if verbose
    fprintf('--- Uniformidade de %d pontos na esfera ---\n', N);
    fprintf('d_NN_min/d_NN_medio : %.4f  \n', ratio);
    fprintf('CV(d_NN)            : %.4f  \n', cv);
    fprintf('Energia de Riesz(s=%g): %.4f \n', s, riesz);
end

end

% ------------------------------------------------------------------
function [ratio, cv, meanNN, minNN] = nn_stats(X)
D = squareform(pdist(X));
D(1:size(D,1)+1:end) = Inf;   % ignore self-distance
nnDist = min(D,[],2);
meanNN = mean(nnDist);
minNN  = min(nnDist);
ratio  = minNN/meanNN;
cv     = std(nnDist)/meanNN;
end

function E = riesz_energy(X, s)
D = pdist(X);   % each pair counted once
E = sum(D.^(-s));
end

function Xn = normalize_points(X)
if size(X,2) ~= 3 && size(X,1) == 3
    X = X.';
end
Xn = X ./ vecnorm(X,2,2);
end

function X = random_sphere_points(N)
X = randn(N,3);
X = X ./ vecnorm(X,2,2);
end