function animate_simulation(x, q_hist, q_d_hist, c_sphere, cells_history, numAgents_history, Param, Nsim)

% === Video settings ===
save_to_mp4 = 0;
frameStep   = 10;                       % must match animation loop
sim_dt      = Param.dTi * frameStep;    % simulated time per frame
videoFPS    = 1 / sim_dt;               % makes video exactly 10 sec

if save_to_mp4
    mp4File = fullfile(pwd,'control_sphere_v6_laypunov_example_3.mp4');
    v = VideoWriter(mp4File,'MPEG-4');
    v.FrameRate = videoFPS;
    open(v);
end

animation = figure('units','normalized','outerposition',[0 0 1 1]);
hold on; grid on; 
axis equal; 
view(3);

% Colors
colors = lines(Param.nD);
% % === Create a label ===
% for iD = 1:Param.nD
%     scatter3(x{iD}(1,1), x{iD}(2,1), x{iD}(3,1), ...
%         50, colors(iD,:), 'x', 'LineWidth', 2, ...
%         'HandleVisibility','off');
% end
% 
% hInit = scatter3(NaN, NaN, NaN, ...
%     50, 'k', 'x', 'LineWidth', 2);
% 
% hProj = scatter3(NaN, NaN, NaN, ...
%     50, 'k', 'd', 'LineWidth', 2);
% 
% hdrone = scatter3(NaN, NaN, NaN, ...
%     50, 'k', 'x');
% 
% hq = scatter3(NaN, NaN, NaN, ...
%     50, 'k', 'o', 'filled');
% 
% hq_d = scatter3(NaN, NaN, NaN, ...
%     50, 'k', '+');
% 
% lgd = legend([hInit, hProj, hdrone, hq, hq_d], ...
%     {'Initial Positions', 'Final Objective', 'drone', ...
%      '$\tilde{q}_i$', '$q_i^\star$'});
% 
% lgd.FontSize = 10; 
% lgd.Interpreter = 'latex';
% lgd.Position = [0.45 0.45 0.05 0.1];  % [x y width height]
% lgd.AutoUpdate = 'off';
% % === End of create a label ===

if strcmp(Param.arena, 'free_world')
    % campos([55.2158  -78.4364   53.1560]);
    % camtarget([2.6397    2.5720    1.7751]);
    % camup([0     0     1]);
    % camva(8.3316);

    % Sphere geometry
    [x_sphere, y_sphere, z_sphere] = sphere;
    x_sphere = x_sphere * Param.sphere_radius;
    y_sphere = y_sphere * Param.sphere_radius;
    z_sphere = z_sphere * Param.sphere_radius;
    
    hSphere = surf(x_sphere, y_sphere, z_sphere);
    hSphere.FaceAlpha = 0.3;
    hSphere.EdgeColor = 'none';
    hSphere.FaceColor = [0.86 0.86 0.86];

    maxAgents = numAgents_history(1);
    
    % Colors
    colors = lines(maxAgents);
    
    % Graphics handles
    for iD = 1:maxAgents
        outer_traj{iD} = plot3(NaN, NaN, NaN,'--', 'Color', colors(iD,:), 'LineWidth', 1);
        inner_traj{iD} = plot3(NaN, NaN, NaN,'-', 'Color', colors(iD,:), 'LineWidth', 1);
        hAgent{iD} = plot3(NaN, NaN, NaN, 'x', 'Color', colors(iD,:), 'MarkerSize', 6, 'MarkerFaceColor', colors(iD,:));
        href{iD} = plot3(NaN, NaN, NaN, 'o', 'Color', 'k', 'MarkerSize', 6, 'MarkerFaceColor', colors(iD,:));
        href_ref{iD} = plot3(NaN, NaN, NaN, '+', 'Color', colors(iD,:), 'MarkerSize', 6, 'MarkerFaceColor', colors(iD,:));
        fib_ref_points_animation{iD} = scatter3(NaN, NaN, NaN, 50, colors(iD,:), 'd', 'LineWidth', 2);
    end
    
    hCenter = plot3(NaN,NaN,NaN,'k','LineWidth',1.5);

    % edges
    hEdges = gobjects(maxAgents,1);
    
    for i = 1:maxAgents
        hEdges(i) = plot3(NaN,NaN,NaN,'Color',colors(i,:),'LineWidth',1.5);
    end
        
    xlabel('x'); ylabel('y'); zlabel('z');
    %title('3D Agent Trajectories');
    
    % Animation loop
    trail_length = 150;
    step = 5;

    for k = 1:step:Nsim
        fprintf("Showing simulation interation: %04d\n", k);
        idx = max(1,k-trail_length):k;

        
        if (k > 1) && (numAgents_history(k) ~= numAgents_history(k-step))
            % Reset all cells
            for i = 1:maxAgents
               set(hEdges(i),'XData',NaN,'YData',NaN,'ZData',NaN);
            end
        end

        numAgents = numAgents_history(k);
        sphere_center = c_sphere(:,k);

        for iD = 1:numAgents
            set(outer_traj{iD}, ...
                'XData', q_hist{iD}(1,idx), ...
                'YData', q_hist{iD}(2,idx), ...
                'ZData', q_hist{iD}(3,idx));
    
            set(inner_traj{iD}, ...
                'XData', x{iD}(1,idx), ...
                'YData', x{iD}(2,idx), ...
                'ZData', x{iD}(3,idx));
    
            set(hAgent{iD}, ...
                'XData', x{iD}(1,k), ...
                'YData', x{iD}(2,k), ...
                'ZData', x{iD}(3,k));
    
            set(href{iD}, 'XData', q_hist{iD}(1, k), ...
                        'YData', q_hist{iD}(2, k), ...
                        'ZData', q_hist{iD}(3, k));
    
            set(href_ref{iD}, ...
                'XData', q_d_hist{iD}(1, k), ...
                'YData', q_d_hist{iD}(2, k), ...
                'ZData', q_d_hist{iD}(3, k));

            verts = cells_history{k,iD};
            edge_pts = [];
            
            for j = 1:size(verts,1)
                p1 = verts(j,:);
                p2 = verts(mod(j,size(verts,1))+1,:);
            
                arc = geodesic_arc(p1,p2,20, Param.sphere_radius, sphere_center');
                edge_pts = [edge_pts; arc];
            end
            
            set(hEdges(iD),...
                'XData',edge_pts(:,1),...
                'YData',edge_pts(:,2),...
                'ZData',edge_pts(:,3));
        end
    
        set(hSphere, ...
            'XData', sphere_center(1)+x_sphere, ...
            'YData', sphere_center(2)+y_sphere, ...
            'ZData', sphere_center(3)+z_sphere);
    
        set(hCenter, ...
            'XData', c_sphere(1,idx), ...
            'YData', c_sphere(2,idx), ...
            'ZData', c_sphere(3,idx));     
    
        drawnow limitrate nocallbacks;
    
        if save_to_mp4
            frame = getframe(animation);
            writeVideo(v,frame);
        else
            %pause(0.01); % adjust speed of animation
        end
    end
end

end
