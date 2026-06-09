function animate_simulation(x, q_hist, q_d_hist, c_sphere, fib_ref_points, Param, Nsim)
%ANIMATE_SIMULATION Summary of this function goes here
%   Detailed explanation goes here

% === Video settings (TIME-ACCURATE) ===
save_to_mp4 = 0;
frameStep   = 10;                       % must match animation loop
sim_dt      = Param.dTi * frameStep;    % simulated time per frame
videoFPS    = 1 / sim_dt;               % makes video exactly 10 sec

animation = figure('units','normalized','outerposition',[0 0 1 1]);
hold on; grid on; axis equal; view(3);

if save_to_mp4
    mp4File = fullfile(pwd,'control_sphere_v6_laypunov_example_2.mp4');
    v = VideoWriter(mp4File,'MPEG-4');
    v.FrameRate = videoFPS;
    open(v);
end

[x_sphere, y_sphere, z_sphere] = sphere;
x_sphere = x_sphere * Param.sphere_radius;
y_sphere = y_sphere * Param.sphere_radius;
z_sphere = z_sphere * Param.sphere_radius;

% Draw reference sphere
hSphere = surf(x_sphere, y_sphere, z_sphere);
hSphere.FaceAlpha = 0.3;
hSphere.EdgeColor = 'none';
hSphere.FaceColor = [0.8627,0.8627,0.8627];

% Setup colors and plot handles for agents
colors = lines(Param.nD);

% === Create a label ===
for iD = 1:1
    scatter3(x{iD}(1,1), x{iD}(2,1), x{iD}(3,1), ...
        50, colors(iD,:), 'x', 'LineWidth', 2, ...
        'HandleVisibility','off');
end

hInit = scatter3(NaN, NaN, NaN, ...
    50, 'k', 'x', 'LineWidth', 2);

hProj = scatter3(NaN, NaN, NaN, ...
    50, 'k', 'd', 'LineWidth', 2);

hdrone = scatter3(NaN, NaN, NaN, ...
    50, 'k', 'x');

hq = scatter3(NaN, NaN, NaN, ...
    50, 'k', 'o', 'filled');

hq_d = scatter3(NaN, NaN, NaN, ...
    50, 'k', '+');

lgd = legend([hInit, hProj, hdrone, hq, hq_d], ...
    {'Initial Positions', 'Final Objective', 'drone', ...
     '$q_i$', '$q_i^\star$'});

lgd.FontSize = 14; 
lgd.Interpreter = 'latex';
lgd.Position = [0.65 0.65 0.1 0.2];  % [x y width height]
lgd.AutoUpdate = 'off';
% === End of create a label ===

for iD = 1:Param.nD
    % Initialize line and marker objects
    outer_loop_traj{iD} = plot3(NaN, NaN, NaN,'--', 'Color', colors(iD,:), 'LineWidth', 1);
    inner_loop_traj{iD} = plot3(NaN, NaN, NaN,'-', 'Color', colors(iD,:), 'LineWidth', 1);
    hAgent{iD} = plot3(NaN, NaN, NaN, 'x', 'Color', colors(iD,:), 'MarkerSize', 6, 'MarkerFaceColor', colors(iD,:));
    href{iD} = plot3(NaN, NaN, NaN, 'o', 'Color', 'k', 'MarkerSize', 6, 'MarkerFaceColor', colors(iD,:));
    href_ref{iD} = plot3(NaN, NaN, NaN, '+', 'Color', colors(iD,:), 'MarkerSize', 6, 'MarkerFaceColor', colors(iD,:));
    fib_ref_points_animation{iD} = scatter3(NaN, NaN, NaN, 50, colors(iD,:), 'd', 'LineWidth', 2);
end

c_sphere_animation = plot3(NaN, NaN, NaN, 'k', 'LineWidth', 1);

xlabel('x'); ylabel('y'); zlabel('z');
%title('3D Trajectories of Agents – Evolution Over Time');

% Time annotation
timeText = text(0,10,10,'t = 0.00 s', ...
    'FontSize',12,'FontWeight','bold');

% Animation loop
for k = 1:frameStep:Nsim
    fprintf("Showing simulation interation: %04d\n", k);

    % Index range: last X steps (or fewer at the beginning)
    %k_start = max(1, k-X);
    k_start = max(1, k-Nsim);
    idx = k_start:k;

    t = (k-1)*Param.dTi;

    for iD = 1:Param.nD
        % Update trajectory up to current time
        set(outer_loop_traj{iD}, 'XData', q_hist{iD}(1, idx), ...
                       'YData', q_hist{iD}(2, idx), ...
                       'ZData', q_hist{iD}(3, idx));
    
        % Update trajectory up to current time
        set(inner_loop_traj{iD}, 'XData', x{iD}(1,idx), ...
                       'YData', x{iD}(2,idx), ...
                       'ZData', x{iD}(3,idx));
        
        % Update agent position marker
        set(hAgent{iD}, 'XData', x{iD}(1, k), ...
                        'YData', x{iD}(2, k), ...
                        'ZData', x{iD}(3, k));
        
        % Update ref position marker
        set(href{iD}, 'XData', q_hist{iD}(1, k), ...
                        'YData', q_hist{iD}(2, k), ...
                        'ZData', q_hist{iD}(3, k));

        set(hSphere, ...
            'XData', c_sphere(1,k)+x_sphere, ...
            'YData', c_sphere(2,k)+y_sphere, ...
            'ZData', c_sphere(3,k)+z_sphere);

        set(c_sphere_animation, ...
            'XData', c_sphere(1,idx), ...
            'YData', c_sphere(2,idx), ...
            'ZData', c_sphere(3,idx));
        
        set(fib_ref_points_animation{iD}, ...
            'XData', c_sphere(1,k) + fib_ref_points(1,iD) ,...
            'YData', c_sphere(2,k) + fib_ref_points(2,iD), ...
            'ZData', c_sphere(3,k) + fib_ref_points(3,iD));
        

        set(href_ref{iD}, ...
            'XData', c_sphere(1,k) + q_d_hist{iD}(1, k), ...
            'YData', c_sphere(2,k) + q_d_hist{iD}(2, k), ...
            'ZData', c_sphere(3,k) + q_d_hist{iD}(3, k));
    end


    % Update time text
    set(timeText,'String',sprintf('t = %.2f s',k*Param.dTi));

    drawnow limitrate nocallbacks;

    if save_to_mp4
        frame = getframe(animation);
        writeVideo(v,frame);
    else
        pause(0.01); % adjust speed of animation
    end
end
end

