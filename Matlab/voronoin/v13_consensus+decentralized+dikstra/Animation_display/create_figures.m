function create_figures(data, Param)    
    if Param.animation.composed == true
        composed_fig(data, Param);
    end

    if Param.animation.radial_hist == true
        radial_hist_plot(data, Param);
    end

    if Param.animation.consensus == true
        consensus_plot(data, Param);
    end

    if Param.animation.save_to_mp4 == true
        mp4File = fullfile(pwd,'control_sphere_v6_laypunov_example_3.mp4');
        v = VideoWriter(mp4File,'MPEG-4');
        v.FrameRate = videoFPS;
        open(v);
    end

    if Param.animation.animate == true
        animate_simulation(data, Param);
    end

    % if Param.animation.create_pdf == true
    %     plot_graphs(data, Param);
    % end
end

