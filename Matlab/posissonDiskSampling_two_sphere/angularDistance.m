function d = angularDistance(p, q)
    d = acos(max(-1,min(1,dot(p,q))));
end
