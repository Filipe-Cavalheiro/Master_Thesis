function animation_cycle(k, data, gfx, Param)
    %ANIMATION_CYCLE Summary of this function goes here
    %   Detailed explanation goes here

    fprintf("Showing simulation interation: %04d\n", k);
    idx = max(1,k-Param.animation.trail_length):k;
    numAgents = data.numAgents_history(k);
    sphere_center = data.c_sphere(:,k);
    maxAgents = data.numAgents_history(1);

    if idx(1) ~= 1 && data.removed_initial_pos == false
        data.removed_initial_pos = true;
        for iD = 1:numAgents
            set(gfx.initial_pos{iD}, 'Visible', 'off');
        end
        lgd = legend([gfx.hdrone, gfx.hq, gfx.hq_d, gfx.hp_d], ...
            {'drone', ...
             '$\tilde{q}_i$', '$q_i^\star$', '$p_i^\star$'}, ...
            'Interpreter', 'latex');
    end

    
    if (k > 1) && (data.numAgents_history(k) ~= data.numAgents_history(k-Param.animation.step))
        % Reset all cells
        for iD = 1:maxAgents
           set(gfx.hEdges(iD),'XData',NaN,'YData',NaN,'ZData',NaN);
           set(gfx.href_ref_final{iD}, 'XData',NaN,'YData',NaN,'ZData',NaN);
        end
    end

    for iD = 1:numAgents
        final_q_d_hist = projectToSphere(data.q_d_hist{iD}(:, k), Param.sphere_radius, sphere_center);


        set(gfx.outer_traj{iD}, ...
            'XData', data.q_hist{iD}(1,idx), ...
            'YData', data.q_hist{iD}(2,idx), ...
            'ZData', data.q_hist{iD}(3,idx));

        set(gfx.inner_traj{iD}, ...
            'XData', data.x{iD}(1,idx), ...
            'YData', data.x{iD}(2,idx), ...
            'ZData', data.x{iD}(3,idx));

        set(gfx.hAgent{iD}, ...
            'XData', data.x{iD}(1,k), ...
            'YData', data.x{iD}(2,k), ...
            'ZData', data.x{iD}(3,k));

        set(gfx.href{iD}, 'XData', data.q_hist{iD}(1, k), ...
                    'YData', data.q_hist{iD}(2, k), ...
                    'ZData', data.q_hist{iD}(3, k));

        set(gfx.href_ref{iD}, ...
            'XData', data.q_d_hist{iD}(1, k), ...
            'YData', data.q_d_hist{iD}(2, k), ...
            'ZData', data.q_d_hist{iD}(3, k));
        
        set(gfx.href_ref_final{iD}, ...
            'XData', final_q_d_hist(1), ...
            'YData', final_q_d_hist(2), ...
            'ZData', final_q_d_hist(3));
        
        verts = data.cells_history{k,iD};
        edge_pts = [];
        
        % for j = 1:size(verts,1)
        %     p1 = verts(j,:);
        %     p2 = verts(mod(j,size(verts,1))+1,:);
        % 
        %     arc = geodesic_arc(p1,p2,20, Param.sphere_radius, sphere_center');
        %     edge_pts = [edge_pts; arc];
        % end
        % 
        % set(gfx.hEdges(iD),...
        %     'XData',edge_pts(:,1),...
        %     'YData',edge_pts(:,2),...
        %     'ZData',edge_pts(:,3));
    end

    set(gfx.hSphere, ...
        'XData', sphere_center(1)+gfx.x_sphere, ...
        'YData', sphere_center(2)+gfx.y_sphere, ...
        'ZData', sphere_center(3)+gfx.z_sphere);

    set(gfx.hCenter, ...
        'XData', data.c_sphere(1,1:k), ...
        'YData', data.c_sphere(2,1:k), ...
        'ZData', data.c_sphere(3,1:k));     

    drawnow limitrate nocallbacks;

    if Param.animation.save_to_mp4
        frame = getframe(animation);
        writeVideo(v,frame);
    else
        % pause(0.01); % adjust speed of animation
    end
end

