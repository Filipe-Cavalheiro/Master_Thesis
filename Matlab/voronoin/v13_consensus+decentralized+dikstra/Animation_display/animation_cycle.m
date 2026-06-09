function animation_cycle(k, data, gfx, Param)
    %ANIMATION_CYCLE Summary of this function goes here
    %   Detailed explanation goes here

    fprintf("Showing simulation interation: %04d\n", k);
    idx = max(1,k-Param.animation.trail_length):k;
    if idx(1) ~= 1 && data.removed_initial_pos == false
        data.removed_initial_pos = true;
        for iD = 1:Param.nD
            set(gfx.initial_pos{iD}, 'Visible', 'off');
        end
        legend([gfx.hdrone, gfx.hp_d], ...
            {'drone', ...
             '$p_i^\star$'}, ...
            'Interpreter', 'latex');
    end
    
    if (k > 1) && (data.numAgents_history(k) ~= data.numAgents_history(k-Param.animation.step))
        % Reset all cells
        for iD = 1:Param.nD
           set(gfx.href_ref_final{iD}, 'XData',NaN,'YData',NaN,'ZData',NaN);
        end
    end

    set(gfx.hp_T, ...
        'XData', data.target_position(1,k), ...
        'YData', data.target_position(2,k), ...
        'ZData', data.target_position(3,k));

    for iD = 1:Param.nD
        %pos = data.chaser{iD}.pos(:,idx);
        set(gfx.inner_traj{iD}, ...
            'XData', data.chaser{iD}.pos(1,idx), ...
            'YData', data.chaser{iD}.pos(2,idx), ...
            'ZData', data.chaser{iD}.pos(3,idx));
        
        set(gfx.hAgent{iD}, ...
            'XData', data.chaser{iD}.pos(1,k), ...
            'YData', data.chaser{iD}.pos(2,k), ...
            'ZData', data.chaser{iD}.pos(3,k));
        
        set(gfx.href_ref_final{iD}, ...
            'XData', data.p_d_hist{iD}(1, k), ...
            'YData', data.p_d_hist{iD}(2, k), ...
            'ZData', data.p_d_hist{iD}(3, k));
    end
        
    set(gfx.hCenter, ...
        'XData', data.target_position(1,1:k), ...
        'YData', data.target_position(2,1:k), ...
        'ZData', data.target_position(3,1:k));     

    drawnow limitrate nocallbacks;

    if Param.animation.save_to_mp4
        frame = getframe(animation);
        writeVideo(v,frame);
    else
        % pause(0.01); % adjust speed of animation
    end
end

