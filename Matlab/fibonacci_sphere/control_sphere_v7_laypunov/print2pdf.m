function print2pdf(outfilename, doprint)
    if exist('doprint','var') && doprint
        fig = gcf;

        % Make figure large on screen
        set(fig, 'Units','centimeters');
        set(fig, 'Position',[1 1 30 20]);   % <-- make this as big as you want

        % Export at high resolution (only affects rasterized content)
        exportgraphics(fig, outfilename, ...
            'ContentType','vector', ...   % keeps text/lines vector
            'Resolution',600);            % high DPI for images
    end
end


