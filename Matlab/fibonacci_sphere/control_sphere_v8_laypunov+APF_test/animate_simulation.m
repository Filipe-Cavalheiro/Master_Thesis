function hfig = animate_simulation(q_hist, fib_ref_points, Param, Nsim)
hfig = figure;
hold on; grid on; 
axis equal; 
view(3);

% Colors
colors = lines(Param.nD);
for iD = 1:Param.nD
    scatter3(q_hist{iD}(1,1), q_hist{iD}(2,1), q_hist{iD}(3,1), ...
        50, colors(iD,:), 'x', 'LineWidth', 2, ...
        'HandleVisibility','off');
end

% Sphere geometry
[x_sphere, y_sphere, z_sphere] = sphere;
x_sphere = x_sphere * Param.sphere_radius;
y_sphere = y_sphere * Param.sphere_radius;
z_sphere = z_sphere * Param.sphere_radius;

hSphere = surf(x_sphere, y_sphere, z_sphere);
hSphere.FaceAlpha = 0.3;
hSphere.EdgeColor = 'b';
hSphere.EdgeAlpha = 0.05;
hSphere.FaceColor = [0.86 0.86 0.86];

% Colors
colors = lines(Param.nD);
% Graphics handles
for iD = 1:Param.nD
    outer_traj{iD} = plot3(NaN, NaN, NaN,'--', 'Color', colors(iD,:), 'LineWidth', 1);
    href{iD} = plot3(NaN, NaN, NaN, 'o', 'Color', 'k', 'MarkerSize', 6, 'MarkerFaceColor', colors(iD,:));
    fib_ref_points_animation{iD} = scatter3(NaN, NaN, NaN, 50, colors(iD,:), 'd', 'LineWidth', 2);
end

hCenter = plot3(NaN,NaN,NaN,'k','LineWidth',1.5);

% Animation loop
trail_length = Nsim;
k = Nsim;
idx = max(1,k-trail_length):k;

for iD = 1:Param.nD
    set(outer_traj{iD}, ...
        'XData', q_hist{iD}(1,idx), ...
        'YData', q_hist{iD}(2,idx), ...
        'ZData', q_hist{iD}(3,idx));

    set(href{iD}, 'XData', q_hist{iD}(1, k), ...
                'YData', q_hist{iD}(2, k), ...
                'ZData', q_hist{iD}(3, k));

    set(fib_ref_points_animation{iD}, ...
        'XData', fib_ref_points(1,iD) ,...
        'YData', fib_ref_points(2,iD), ...
        'ZData', fib_ref_points(3,iD));
end
   
drawnow;
pause(0.5); % adjust speed of animation
end



