function composed_fig(data, Param)
    figure;
    hold on; 
    grid on; 
    axis equal;
    view(3);

    target_position = data.target_position;
    chaser = data.chaser;
    
    % data.colors for chasers
    steps = [50 300 600 1000 1500 2000 3000];
    steps(steps> Param.Nsim) = []; % remove values above simulation time
    
    % === Create a label ===
    hdrone = scatter3(NaN, NaN, NaN, ...
        50, 'k', 'x');

    htarget_path = plot3(NaN, NaN, NaN, 'k--');
    
    htarget_pos= scatter3(NaN, NaN, NaN, 'k', 'o');

    hpath = plot3(NaN, NaN, NaN, 'k');

    hfinal_pos = scatter3(NaN, NaN, NaN, 'k', 'd');

    lgd = legend([hdrone, hpath, htarget_pos, htarget_path, hfinal_pos], ...
        {'$p_i$' ,'$p_i$ path', '$p_T$', '$p_T$ path',...
         '$p^\star$'}, 'Interpreter','latex');

    lgd.FontSize = 10; 
    lgd.Interpreter = 'latex';
    lgd.Position = [0.5825    0.7677    0.0752    0.1113];  % [x y width height]
    lgd.AutoUpdate = 'off';
    % === End of create a label ===
    
    %plot3(Param.ref_target_traj(1,:), Param.ref_target_traj(2,:), Param.ref_target_traj(3,:), 'k--');
    
    for k = steps
        idx = max(1,k-Param.animation.trail_length):k;
        % --- target position at timestep k ---
        scatter3(target_position(1,k), ...
                 target_position(2,k), ...
                 target_position(3,k), ...
                 'k', ...
                 'o');

        text(target_position(1,k)-2,...
            target_position(2,k)-2,...
            target_position(3,k)+8,...
            ['   ' num2str(k/100) ' sec'],...
            'HorizontalAlignment','left',...
            'FontSize',8);

        % --- chaser positions at timestep k ---
        for iD = 1:Param.nD
            plot3(chaser{iD}.pos(1,idx), ...
                  chaser{iD}.pos(2,idx), ...
                  chaser{iD}.pos(3,idx), ...
                  'Color', data.colors(iD,:));

            scatter3(chaser{iD}.pos(1,k), ...
                     chaser{iD}.pos(2,k), ...
                     chaser{iD}.pos(3,k), ...
                     'x', 'MarkerEdgeColor', data.colors(iD,:));
        end
    end
    print2pdf(sprintf("composed_%d_agents", Param.nD), 1);
end

