clear; clc; close all;

% Define the points in 3D space
point1 = [2, 3, 8];  % Example point 1
point2 = [-1, 2, 6]; % Example point 2

% Define the sphere radius and center
radius = 5;
center = [0, 0, 0];

points_on_sphere = projectToSphere([point1; point2], radius);

A_point = points_on_sphere(1, :);
B_point = points_on_sphere(2, :);

% Ensure normalized to sphere radius R
A_point = (A_point / norm(A_point)) * radius;
B_point = (B_point / norm(B_point)) * radius;

% Direction vector from A to B (parallel direction)
AB_direction =  A_point - B_point;

% Scale the direction vector for the new point
new_points_on_sphere = A_point + AB_direction * 5;  % Move along the same direction by a scalar factor

% Ensure the new point is on the sphere by normalizing
%new_points_on_sphere = (new_points_on_sphere / norm(new_points_on_sphere)) * radius;

% Create a figure
figure;

% Plot the sphere
[X, Y, Z] = sphere(50); % Create sphere data with 50 points in each direction
hSphere = surf(X*radius,Y*radius,Z*radius);
hSphere.FaceAlpha = 0.3;   % Set transparency (0 = fully transparent, 1 = opaque)
hSphere.EdgeColor = 'none'; % Optional: remove mesh lines for smoother look
hold on;

% Plot the two points
plot3(point1(1), point1(2), point1(3), 'ro', 'MarkerFaceColor', 'r'); % Point 1 in red
plot3(point2(1), point2(2), point2(3), 'bo', 'MarkerFaceColor', 'b'); % Point 2 in blue
plot3(points_on_sphere(:,1), points_on_sphere(:,2), points_on_sphere(:,3), 'bx', 'MarkerFaceColor', 'b'); % Projected Point 1 in blue
plot3(new_points_on_sphere(1), new_points_on_sphere(2), new_points_on_sphere(3), 'gx', 'MarkerFaceColor', 'g'); % New Point in green

quiver3(A_point(1), A_point(2), A_point(3), AB_direction(1), AB_direction(2), AB_direction(3), 1.5, 'k--', 'LineWidth', 1.5, 'ShowArrowHead','on');


% Add text labels for the points in 3D
text(point1(1), point1(2), point1(3), ' A', 'FontSize', 12, 'Color', 'r');
text(point2(1), point2(2), point2(3), ' B', 'FontSize', 12, 'Color', 'b');
text(points_on_sphere(1,1), points_on_sphere(1,2), points_on_sphere(1,3), ' A (on sphere)', 'FontSize', 12, 'Color', 'r');
text(points_on_sphere(2,1), points_on_sphere(2,2), points_on_sphere(2,3), ' B (on sphere)', 'FontSize', 12, 'Color', 'b');
text(new_points_on_sphere(1), new_points_on_sphere(2), new_points_on_sphere(3), ' New A', 'FontSize', 12, 'Color', 'g');

% Labels and title
xlabel('X');
ylabel('Y');
zlabel('Z');
title('3D Plot of Two Points and a Sphere');

% Set axis limits for better visibility
axis equal;

grid on;
hold off;
