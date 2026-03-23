clc; clear; close all;

% --- Surface definition ---
surface = @(x,y) x.^2 + y.^2;

% --- Initial random sampling ---
n = 25;
pts = 2 * rand(n,2) - 1;

% --- Lloyd iterations ---
numIter = 1000;   % increase for smoother distribution

for it = 1:numIter
    
    % Voronoi diagram
    [V, C] = voronoin(pts);
    
    newPts = zeros(size(pts));
    
    for i = 1:length(C)
        cellVerts = C{i};
        
        % Skip infinite regions
        if all(cellVerts ~= 1)
            verts = V(cellVerts, :);
            
            % Compute centroid of polygon
            centroid = mean(verts, 1);
            
            % Keep inside domain [-1,1]
            centroid = max(min(centroid,1),-1);
            
            newPts(i,:) = centroid;
        else
            % Keep old point if region is infinite
            newPts(i,:) = pts(i,:);
        end
    end
    
    pts = newPts;
end

% Extract relaxed points
x = pts(:,1);
y = pts(:,2);
z = surface(x,y);

% --- Delaunay triangulation ---
tri = delaunay(x,y);

% --- Plot ---
figure;
trisurf(tri, x, y, z, ...
    'FaceColor', [0.8 0.9 1], ...
    'EdgeColor', 'k');

xlabel('x'); ylabel('y'); zlabel('z');
title('Lloyd Relaxed Delaunay Triangulation of z = x^2 + y^2');

axis equal;
view(3);
camlight;
lighting phong;