classdef MeshVertex
    % Stores per-vertex data for shortest-path on a triangle mesh graph.
    properties
        id              (1,1) double
        pos             (1,3) double
        dist            (1,1) double = inf
        neigh           (:,1) double = zeros(0,1)   % neighbor vertex indices
        neighW          (:,1) double = zeros(0,1)   % corresponding edge lengths
        prev            (1,1) double = 0            % predecessor (0 = none)
        origin_site     (1,1) double = -1   % origin vertex indice
    end
    methods
        function obj = MeshVertex(id, pos)
            if nargin > 0
                obj.id = id;
                obj.pos = pos(:).';
            end
        end
    end
end