function results = sphere_uniformity(X, varargin)
%SPHERE_UNIFORMITY  Assess how uniformly points are spaced on a sphere.
%
%   results = sphere_uniformity(X) computes homogeneity metrics for a point
%   set X (N-by-3 or 3-by-N).
%
%   Metrics:
%     - ratio_min_mean : d_NN,min / d_NN,mean  (closer to 1 = more uniform)
%     - cv_nn          : std(d_NN)/mean(d_NN)  (lower = more uniform)
%     - riesz_energy   : sum_{i<j} 1/d_ij^s    (on the unit sphere)
%     - mean_nn, min_nn, max_nn : nearest-neighbour chordal distances,
%                        scaled back to physical units ([m])
%
%   Name-value options:
%     'Center'  1x3 or 3x1 sphere centre (default [0 0 0]). Points are
%               translated by -Center before being projected on the sphere.
%     'Radius'  sphere radius used to scale distances back to metres
%               (default 1 = unit sphere).
%     'RieszS'  Riesz exponent (default 1)
%     'Verbose' print a summary (default false)

p = inputParser;
addParameter(p,'Center',[0 0 0]);
addParameter(p,'Radius',1);
addParameter(p,'RieszS',1);
addParameter(p,'Verbose',false);
parse(p,varargin{:});
c       = p.Results.Center(:).';
R       = p.Results.Radius;
s       = p.Results.RieszS;
verbose = p.Results.Verbose;

if size(X,2) ~= 3 && size(X,1) == 3
    X = X.';
end
X = X - c;                      % centre on the sphere
X = X ./ vecnorm(X,2,2);        % project onto the unit sphere
N = size(X,1);

[ratio, cv, meanNN, minNN, maxNN] = nn_stats(X);

results.N              = N;
results.ratio_min_mean = ratio;
results.cv_nn          = cv;
results.mean_nn        = R*meanNN;
results.min_nn         = R*minNN;
results.max_nn         = R*maxNN;
results.riesz_energy   = riesz_energy(X, s);

if verbose
    fprintf('--- Uniformity of %d points on the sphere ---\n', N);
    fprintf('d_NN_min/d_NN_mean  : %.4f\n', ratio);
    fprintf('CV(d_NN)            : %.4f\n', cv);
    fprintf('Riesz energy (s=%g) : %.4f\n', s, results.riesz_energy);
end
end

% ------------------------------------------------------------------
function [ratio, cv, meanNN, minNN, maxNN] = nn_stats(X)
D = squareform(pdist(X));
D(1:size(D,1)+1:end) = Inf;     % ignore self-distance
nnDist = min(D,[],2);
meanNN = mean(nnDist);
minNN  = min(nnDist);
maxNN  = max(nnDist);
ratio  = minNN/meanNN;
cv     = std(nnDist)/meanNN;
end

function E = riesz_energy(X, s)
D = pdist(X);                   % each pair counted once
E = sum(D.^(-s));
end