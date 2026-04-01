function [uv, coeff] = flatten_surface_pca(XYZ)
    % XYZ: Nx3 matrix of 3D points
    % uv: Nx2 matrix of 2D coordinates
    % coeff: 3x3 PCA transformation matrix
    
    % Center the data
    XYZ_centered = XYZ - mean(XYZ, 1);
    
    % PCA (eigen decomposition of covariance matrix)
    [coeff, ~, ~] = pca(XYZ_centered);
    
    % Project onto first 2 principal components
    uv = XYZ_centered * coeff(:,1:2);
end

% Generate or load 3D object surface
[X, Y, Z] = sphere(40);  % Example: Sphere
XYZ = [X(:), Y(:), Z(:)];

% Flatten using PCA
[uv, coeff] = flatten_surface_pca(XYZ);

% Plot
figure;
subplot(1,2,1)
surf(X, Y, Z)
title('Original 3D Surface')
axis equal

subplot(1,2,2)
scatter(uv(:,1), uv(:,2), 10, Z(:), 'filled')
title('Flattened 2D Projection (PCA)')
axis equal
colorbar
