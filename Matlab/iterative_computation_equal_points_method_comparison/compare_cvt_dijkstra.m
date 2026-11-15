% COMPARE_CVT_DIJKSTRA  Lloyd (convex hull) vs Lloyd (half plane) vs Lloyd
% (Dijkstra on a triangle mesh), all on the unit two-sphere and under the
% conditions of the iterative comparison of the thesis annex:
%   30 sites, 100 runs, at most 1000 iterations, tol = 5e-3 on the largest
%   site displacement between two iterations, random initial sites.
%
% The convex-hull and half-plane loops are the ones of
%   voronoin/Learning/convex_hull/S2_voronoi/S2_voronoin.m
%   voronoin/Learning/halth_plane_method/S2/voronoi_half_plane.m
% and the Dijkstra CVT uses the functions of voronoin/v14/VoronoiCVT
% (buildVertexObjects, dijkstraMeshHeap), with a Lloyd step that updates
% every site (the v14 one only updates the site of the calling agent).
%
% Usage:  compare_cvt_dijkstra            full benchmark (100 runs)
%         nRuns = 5; compare_cvt_dijkstra quick test
%         replot = true; compare_cvt_dijkstra   figures from the saved .mat
%
% Output: cvt_dijkstra_comparison.mat, PDF/cvt_dijkstra_*.pdf

if ~exist('nRuns', 'var'), nRuns = 100; end
if ~exist('replot', 'var'), replot = false; end
clearvars -except nRuns replot; close all; clc;

here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, '..', 'voronoin', 'v14', 'VoronoiCVT'));

numAgents  = 30;
iterations = 1000;
tol        = 5e-3;
meshSizes  = [1001 3001 10001];   % vertices of the Fibonacci sphere mesh
meshRuns   = min(nRuns, [100 100 20]);  % the finest mesh takes ~40 s per run
refMesh    = 3001;                % resolution used in the main comparison
rng(0);

% Same initial sites for the three methods
init = cell(nRuns, 1);
for run = 1:nRuns
    a = randn(numAgents, 3);
    init{run} = a ./ vecnorm(a, 2, 2);
end

if exist('replot', 'var') && replot
    load(fullfile(here, 'cvt_dijkstra_comparison.mat'));
    makeFigures(here, res, meshSizes, refMesh);
    return
end

%% Lloyd (convex hull)
res.ch = emptyResult(nRuns);
for run = 1:nRuns
    agents = init{run};
    tic;
    for k = 1:iterations
        prevAgents = agents;
        K = convhull(agents(:,1), agents(:,2), agents(:,3));
        numFaces = size(K, 1);
        V = zeros(numFaces, 3);
        for i = 1:numFaces
            p1 = agents(K(i,1),:); p2 = agents(K(i,2),:); p3 = agents(K(i,3),:);
            n = cross(p2 - p1, p3 - p1);
            V(i,:) = n / norm(n);
        end
        newAgents = zeros(size(agents));
        for i = 1:numAgents
            verts = V(any(K == i, 2), :);
            if isempty(verts)
                newAgents(i,:) = agents(i,:);
                continue
            end
            c = mean(verts, 1);
            newAgents(i,:) = c / norm(c);
        end
        agents = newAgents;
        if max(vecnorm(agents - prevAgents, 2, 2)) < tol
            res.ch.iters(run) = k;
            res.ch.converged(run) = true;
            break
        end
    end
    res.ch.times(run) = toc;
    res.ch = storeQuality(res.ch, run, agents);
end
report('Lloyd (convex hull)', res.ch);

%% Lloyd (half plane)
[X, Y, Z] = sphere(50);
P = [X(:), Y(:), Z(:)];
[nr, nc] = size(X);
[I, J] = ndgrid(1:nr-1, 1:nc-1);
idxA = sub2ind(size(X), I(:),   J(:));
idxB = sub2ind(size(X), I(:)+1, J(:));
idxC = sub2ind(size(X), I(:),   J(:)+1);
idxD = sub2ind(size(X), I(:)+1, J(:)+1);
A0 = P(idxA,:); B0 = P(idxB,:); C0 = P(idxC,:); D0 = P(idxD,:);

res.hp = emptyResult(nRuns);
for run = 1:nRuns
    agents = init{run};
    tic;
    for k = 1:iterations
        new_agents = zeros(numAgents, 3);
        for iD = 1:numAgents
            current_agent = agents(iD,:);
            other_agents  = agents([1:iD-1, iD+1:numAgents], :);
            Nother = size(other_agents, 1);
            p1 = repmat(current_agent, Nother, 1);
            p2 = other_agents;
            cosTheta = min(max(sum(p1 .* p2, 2), -1), 1);
            theta = acos(cosTheta);
            m = (sin(0.5*theta).*p1 + sin(0.5*theta).*p2) ./ sin(theta);
            n = cross(p1, p2, 2);
            n = n ./ vecnorm(n, 2, 2);
            tangent = cross(n, m, 2);
            tangent = tangent ./ vecnorm(tangent, 2, 2);
            offsets = sum(m .* tangent, 2);
            mask = all(P * tangent' < offsets', 2);
            validCell = mask(idxA) & mask(idxB) & mask(idxC) & mask(idxD);
            area1 = vecnorm(cross(B0 - A0, C0 - A0, 2), 2, 2) / 2;
            centroid1 = (A0 + B0 + C0) / 3;
            area2 = vecnorm(cross(C0 - D0, B0 - D0, 2), 2, 2) / 2;
            centroid2 = (D0 + C0 + B0) / 3;
            area1(~validCell) = 0;
            area2(~validCell) = 0;
            centroid = sum(area1.*centroid1 + area2.*centroid2, 1) / (sum(area1) + sum(area2));
            new_agents(iD,:) = centroid ./ norm(centroid);
        end
        moved = max(vecnorm(new_agents - agents, 2, 2));
        agents = new_agents;
        if moved < tol
            res.hp.iters(run) = k;
            res.hp.converged(run) = true;
            break
        end
    end
    res.hp.times(run) = toc;
    res.hp = storeQuality(res.hp, run, agents);
end
report('Lloyd (half plane)', res.hp);

%% Lloyd (Dijkstra), one result per mesh resolution
res.dj = cell(numel(meshSizes), 1);
meshEdge = zeros(numel(meshSizes), 1);
for mIdx = 1:numel(meshSizes)
    [vertices, faces] = fibonacciSphereMesh(meshSizes(mIdx));
    E = unique(sort([faces(:,[1 2]); faces(:,[2 3]); faces(:,[3 1])], 2), 'rows');
    meshEdge(mIdx) = mean(vecnorm(vertices(E(:,1),:) - vertices(E(:,2),:), 2, 2));

    nR = meshRuns(mIdx);
    r = emptyResult(nR);
    r.cycle = zeros(1, nR);         % period of the limit cycle (0 = none)
    for run = 1:nR
        % initial sites: mesh vertices closest to the random points
        sites = zeros(1, numAgents);
        for s = 1:numAgents
            d = vecnorm(vertices - init{run}(s,:), 2, 2);
            d(sites(1:s-1)) = inf;
            [~, sites(s)] = min(d);
        end
        seen = containers.Map('KeyType', 'char', 'ValueType', 'double');
        tic;
        for k = 1:iterations
            Vobj = buildVertexObjects(vertices, faces);
            Vobj = dijkstraMeshHeap(Vobj, sites);
            newSites = lloydStepAllSites(vertices, faces, Vobj, sites);
            moved = max(vecnorm(vertices(newSites,:) - vertices(sites,:), 2, 2));
            sites = newSites;
            if moved < tol
                r.iters(run) = k;
                r.converged(run) = true;
                break
            end
            % A discrete Lloyd iteration can enter a limit cycle; once a
            % configuration repeats, the remaining iterations are known,
            % so the run is stopped and counted as not converged.
            key = sprintf('%d,', sites);
            if isKey(seen, key)
                r.cycle(run) = k - seen(key);
                r.iters(run) = k;
                break
            end
            seen(key) = k;
        end
        r.times(run) = toc;
        r.time_per_iter(run) = r.times(run) / k;
        r = storeQuality(r, run, vertices(sites,:));
        fprintf('  mesh %5d run %3d/%3d: %2d it, %.2f s, conv %d, cycle %d\n', ...
            meshSizes(mIdx), run, nR, k, r.times(run), r.converged(run), r.cycle(run));
    end
    report(sprintf('Lloyd (Dijkstra, %d vertices)', meshSizes(mIdx)), r);
    fprintf('  mean edge %.4f, runs ending in a limit cycle: %d\n', meshEdge(mIdx), nnz(r.cycle));
    res.dj{mIdx} = r;
end

save(fullfile(here, 'cvt_dijkstra_comparison.mat'), 'res', 'meshSizes', 'meshRuns', 'meshEdge', ...
    'numAgents', 'iterations', 'tol', 'nRuns', 'refMesh');

%% Figures
makeFigures(here, res, meshSizes, refMesh);

%% ---------------------------------------------------------------- functions
function r = emptyResult(nRuns)
    r.times = zeros(1, nRuns);
    r.iters = zeros(1, nRuns);
    r.converged = false(1, nRuns);
    r.ratio = zeros(1, nRuns);
    r.cv = zeros(1, nRuns);
    r.riesz = zeros(1, nRuns);
    r.final = cell(1, nRuns);
end

function r = storeQuality(r, run, X)
    D = squareform(pdist(X));
    D(1:size(D,1)+1:end) = Inf;
    nn = min(D, [], 2);
    r.ratio(run) = min(nn) / mean(nn);
    r.cv(run)    = std(nn) / mean(nn);
    r.riesz(run) = sum(pdist(X).^(-1));
    r.final{run} = X;
end

function report(name, r)
    c = r.converged;
    fprintf('\n%s\n', name);
    fprintf('  mean run time %.4f s (median %.4f, std %.4f)\n', mean(r.times), median(r.times), std(r.times));
    fprintf('  converged %d / %d, mean iterations (converged runs) %.3f\n', nnz(c), numel(c), mean(r.iters(c)));
    fprintf('  ratio %.4f +- %.4f | CV %.4f +- %.4f | Riesz %.2f +- %.2f\n', ...
        mean(r.ratio), std(r.ratio), mean(r.cv), std(r.cv), mean(r.riesz), std(r.riesz));
    if isfield(r, 'time_per_iter')
        fprintf('  mean time per iteration %.4f s\n', mean(r.time_per_iter));
    end
end

function [vertices, faces] = fibonacciSphereMesh(P)
    % Same mesh as generateMesh('sphere') of the Dijkstra demo.
    phi_golden = (1 + sqrt(5)) / 2;
    i = (0:P-1)';
    phi = 2*pi*i/phi_golden;
    theta = acos(1 - 2*(i+0.5)/P);
    vertices = [cos(theta), sin(theta).*sin(phi), sin(theta).*cos(phi)];
    faces = convhull(vertices(:,1), vertices(:,2), vertices(:,3));
end

function newSites = lloydStepAllSites(vertices, faces, V, sites)
    % Lloyd step of the thesis (area-weighted centroid of the triangles that
    % lie entirely in a cell, snapped to the closest free vertex), applied
    % to every site.
    orig = [V.origin_site]';
    s = orig(faces);
    inside = s(:,1) == s(:,2) & s(:,2) == s(:,3);
    p1 = vertices(faces(inside,1),:);
    p2 = vertices(faces(inside,2),:);
    p3 = vertices(faces(inside,3),:);
    A = 0.5 * vecnorm(cross(p2 - p1, p3 - p1, 2), 2, 2);
    c = (p1 + p2 + p3) / 3;
    owner = s(inside, 1);

    newSites = sites;
    for k = 1:numel(sites)
        sel = owner == sites(k);
        areaSum = sum(A(sel));
        if areaSum == 0, continue; end
        centroid = sum(A(sel) .* c(sel,:), 1) / areaSum;
        d = vecnorm(vertices - centroid, 2, 2);
        d(newSites([1:k-1, k+1:end])) = inf;   % no two sites on one vertex
        [~, newSites(k)] = min(d);
    end
end

function makeFigures(here, res, meshSizes, refMesh)
    nRuns = numel(res.ch.times);
    ref = res.dj{meshSizes == refMesh};
    figure;
    boxplot([res.ch.times(:); res.hp.times(:); ref.times(:)], ...
        [repmat("Lloyd (convex hull)", nRuns, 1); repmat("Lloyd (half plane)", nRuns, 1); ...
         repmat("Lloyd (Dijkstra)", nRuns, 1)]);
    set(gca, 'YScale', 'log', 'YLim', [5e-3 100], 'YTick', [0.01 0.1 1 10 100], ...
        'YTickLabel', {'0.01', '0.1', '1', '10', '100'});
    xlabel('Analysis Method'); ylabel('Execution Time [Sec]'); grid on;
    exportFig(here, 'cvt_dijkstra_run_time');

    figure;
    boxplot([res.ch.cv(:); res.hp.cv(:); ref.cv(:)], ...
        [repmat("Lloyd (convex hull)", nRuns, 1); repmat("Lloyd (half plane)", nRuns, 1); ...
         repmat("Lloyd (Dijkstra)", nRuns, 1)]);
    xlabel('Analysis Method'); ylabel('CV(d_{NN})'); grid on;
    exportFig(here, 'cvt_dijkstra_cv');
end

function exportFig(here, name)
    pdfdir = fullfile(here, 'PDF');
    if ~exist(pdfdir, 'dir'), mkdir(pdfdir); end
    set(gcf, 'Units', 'centimeters', 'Position', [1 1 30 20]);
    exportgraphics(gcf, fullfile(pdfdir, [name '.pdf']), 'ContentType', 'vector');
end
