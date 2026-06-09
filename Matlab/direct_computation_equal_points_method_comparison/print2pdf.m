function print2pdf(outfilename, doprint)
    if nargin > 1 && doprint
        exportgraphics(gcf, outfilename, ...
            'ContentType','vector', ...
            'BackgroundColor','none');
    end
end