function [points,iterCount] = poissonDiskSampling(width, height, r, k)
    % Bridson's Poisson Disk Sampling in 2D
    % width, height: dimensions of the domain
    % r: minimum distance between points
    % k: number of candidate points to try per active point (default: 30)

    if nargin < 4
        k = 30;
    end

    % Cell size
    cellSize = r / sqrt(2);

    % Grid dimensions
    gridWidth = ceil(width / cellSize);
    gridHeight = ceil(height / cellSize);
    grid = cell(gridHeight, gridWidth);

    % Lists
    points = [];
    activeList = [];

    % Initial random point
    x0 = rand * width;
    y0 = rand * height;

    points(end+1, :) = [x0, y0];
    activeList(end+1, :) = [x0, y0];

    % Add to grid
    gx = floor(x0 / cellSize) + 1;
    gy = floor(y0 / cellSize) + 1;
    grid{gy, gx} = [x0, y0];
    iterCount = 0;

    while ~isempty(activeList)
        iterCount = iterCount + 1;
        % Randomly pick a point from the active list
        idx = randi(size(activeList, 1));
        point = activeList(idx, :);
        found = false;

        for i = 1:k
            % Generate random point in the annulus (r to 2r)
            theta = 2 * pi * rand;
            radius = r * (1 + rand);
            newPoint = point + radius * [cos(theta), sin(theta)];

            % Check bounds
            if newPoint(1) < 0 || newPoint(1) >= width || ...
               newPoint(2) < 0 || newPoint(2) >= height
                continue;
            end

            % Check if point is valid
            gx = floor(newPoint(1) / cellSize) + 1;
            gy = floor(newPoint(2) / cellSize) + 1;

            ok = true;
            for iY = max(gy-2, 1):min(gy+2, gridHeight)
                for iX = max(gx-2, 1):min(gx+2, gridWidth)
                    neighbor = grid{iY, iX};
                    if ~isempty(neighbor)
                        if norm(newPoint - neighbor) < r
                            ok = false;
                            break;
                        end
                    end
                end
                if ~ok
                    break;
                end
            end

            if ok
                points(end+1, :) = newPoint;
                activeList(end+1, :) = newPoint;
                grid{gy, gx} = newPoint;
                found = true;
                break;
            end
        end

        if ~found
            activeList(idx, :) = [];
        end
    end
end
