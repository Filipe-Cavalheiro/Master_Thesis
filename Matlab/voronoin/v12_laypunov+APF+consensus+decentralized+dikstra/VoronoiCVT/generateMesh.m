function [vertices, faces] = generateMesh(shapeType, res, Param)

    switch lower(shapeType)

        case 'circle'
            [theta, radius] = meshgrid(linspace(0, 2*pi, res), linspace(0, 1, res));
            x = radius .* cos(theta);
            y = radius .* sin(theta);
            z = zeros(size(x));

            [faces, vertices] = surf2patch(x, y, z, 'triangles');

        case 'sphere'
            [X,Y,Z] = sphere(res);
            [faces, vertices] = surf2patch(X,Y,Z,'triangles');

            % normalize (important if using sphere())
            vertices = Param.sphere_radius*(vertices ./ vecnorm(vertices,2,2));

        case 'ellipsoid'
            a = 1; b = 0.6; c = 0.4; % axis scales

            [X,Y,Z] = sphere(res);
            X = Param.sphere_radius* a * X;
            Y = Param.sphere_radius* b * Y;
            Z = Param.sphere_radius* c * Z;

            [faces, vertices] = surf2patch(X,Y,Z,'triangles');

        case 'paraboloid'
            [r, theta] = meshgrid(linspace(0,1,res), linspace(0,2*pi,res));

            x = r .* cos(theta);
            y = r .* sin(theta);
            z = r.^2;   % paraboloid z = r^2

            [faces, vertices] = surf2patch(x,y,z,'triangles');

        otherwise
            error('Unknown shape type');
    end

    % --- Weld duplicates (important for all shapes)
    tol = 1e-10;
    [verticesW, ~, ic] = uniquetol(vertices, tol, 'ByRows', true);

    facesW = ic(faces);

    % Remove degenerate faces
    deg = facesW(:,1)==facesW(:,2) | facesW(:,2)==facesW(:,3) | facesW(:,1)==facesW(:,3);
    facesW(deg,:) = [];

    vertices = verticesW;
    faces    = facesW;
end