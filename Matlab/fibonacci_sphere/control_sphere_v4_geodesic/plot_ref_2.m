clear; clc; close all;

% Define the points in 3D space
point1 = [2, 3, 8];  % Example point 1
point2 = [-1, 2, 6]; % Example point 2

% Define the sphere
radius = 5;
center = [0, 0, 0];

% Project points to the sphere
points_on_sphere = projectToSphere([point1; point2], radius);
A_point = points_on_sphere(1, :);
B_point = points_on_sphere(2, :);

% A_point = point1;
% B_point = point2;

A_norm = A_point / norm(A_point);
B_norm = B_point / norm(B_point);

% Compute tangent direction (great-circle direction)
tangent_dir = B_norm - dot(B_norm, A_norm) * A_norm;  % projection of B onto tangent plane at A
tangent_dir = -tangent_dir / norm(tangent_dir);        % direction away from B on the sphere

% Compute projected point of B onto tangent plane at A
B_proj_on_plane = B_norm - dot(B_norm, A_norm) * A_norm; % projection vector
B_proj_on_plane = A_point + B_proj_on_plane * radius;    % move it to plane position near A

% Define a small step along the tangent direction
step_size = 5; % just for visualization
new_A = A_point + tangent_dir * step_size; 

new_A_proj = (new_A / norm(new_A)) * radius;

% Create a figure
figure;
hold on; axis equal; grid on;

% Plot the sphere
[X, Y, Z] = sphere(60);
surf(radius*X, radius*Y, radius*Z, 'FaceAlpha', 0.2, 'EdgeColor', 'none');

% Plot the points
plot3(A_point(1), A_point(2), A_point(3), 'ro', 'MarkerFaceColor', 'r', 'DisplayName','A Point');
plot3(B_point(1), B_point(2), B_point(3), 'bo', 'MarkerFaceColor', 'b', 'DisplayName','B Point');
plot3(B_proj_on_plane(1), B_proj_on_plane(2), B_proj_on_plane(3), 'ko', 'MarkerFaceColor', 'k', 'DisplayName','B projected on plane');
plot3(new_A(1), new_A(2), new_A(3), 'go', 'MarkerFaceColor', 'g', 'DisplayName','New A');
plot3(new_A_proj(1), new_A_proj(2), new_A_proj(3), 'go', 'MarkerFaceColor', 'g', 'DisplayName','New A Proj');

% Draw the tangent plane at A
plane_size = 2;
% Create orthogonal basis vectors on the tangent plane
if abs(A_norm(3)) < 0.9
    temp = [0 0 1];
else
    temp = [1 0 0];
end
v1 = cross(A_norm, temp); v1 = v1 / norm(v1);
v2 = cross(A_norm, v1);   v2 = v2 / norm(v2);

% Plane grid
[s1, s2] = meshgrid(linspace(-plane_size, plane_size, 10));
plane_points = A_point' + v1'*s1(:)' + v2'*s2(:)';
X_plane = reshape(plane_points(1,:), size(s1));
Y_plane = reshape(plane_points(2,:), size(s1));
Z_plane = reshape(plane_points(3,:), size(s1));
surf(X_plane, Y_plane, Z_plane, 'FaceAlpha', 0.3, 'FaceColor', [0.8 0.8 0.8], 'EdgeColor','none', 'DisplayName','Tangent plane');

% Draw tangent direction arrow
quiver3(A_point(1), A_point(2), A_point(3), tangent_dir(1), tangent_dir(2), tangent_dir(3), 2, 'g', 'LineWidth', 2, 'DisplayName','Tangent direction');
quiver3(A_point(1), A_point(2), A_point(3), -A_norm(1), -A_norm(2), -A_norm(3), 1.5, 'k--', 'LineWidth', 1.5, 'DisplayName','Normal to plane');

% Labels
text(A_point(1), A_point(2), A_point(3), '  A (on sphere)', 'Color','r','FontSize',10);
text(B_point(1), B_point(2), B_point(3), '  B (on sphere)', 'Color','b','FontSize',10);
text(B_proj_on_plane(1), B_proj_on_plane(2), B_proj_on_plane(3), '  B proj on plane', 'Color','k','FontSize',10);
text(new_A(1), new_A(2), new_A(3), '  Moved A', 'Color','g','FontSize',10);
text(new_A_proj(1), new_A_proj(2), new_A_proj(3), ' New A Proj', 'Color','g','FontSize',10);

xlabel('X'); ylabel('Y'); zlabel('Z');
title('Tangent Plane and Direction of Repulsion on a Sphere');
legend('show');
view(3);
hold off;

% Helper function: project points to sphere
function pts_on_sphere = projectToSphere(points, R)
    norms = sqrt(sum(points.^2, 2));
    pts_on_sphere = R * (points ./ norms);
end
