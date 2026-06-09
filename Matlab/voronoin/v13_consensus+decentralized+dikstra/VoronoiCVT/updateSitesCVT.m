function newsite = updateSitesCVT(vertices, faces, V, sites)

    targetSite = sites(1);
    orig = [V.origin_site]';
    
    centroid = [0, 0, 0];
    areaSum  = 0;
    
    for f = 1:size(faces,1)
        tri = faces(f,:);
        s = orig(tri);
    
        % only triangles belonging to site(1)
        if s(1)==s(2) && s(2)==s(3) && s(1)==targetSite
    
            p1 = vertices(tri(1),:);
            p2 = vertices(tri(2),:);
            p3 = vertices(tri(3),:);
    
            % centroid
            c = (p1 + p2 + p3)/3;
    
            % area
            A = 0.5 * norm(cross(p2-p1, p3-p1));
    
            centroid = centroid + A * c;
            areaSum  = areaSum + A;
        end
    end
        
    if areaSum == 0
        newsite = targetSite;
        return;
    end

    c = centroid / areaSum;

    % snap to closest vertex
    d = vecnorm(vertices - c, 2, 2);
    [~, newsite] = min(d);
end