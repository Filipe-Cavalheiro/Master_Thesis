function [vertices, faces] = loadObject(state, Param)

    name = Param.objList{state.obj_index};

    if state.obj_index < 5
        [vertices, faces] = generateMesh(name, Param.mesh_resolution, Param);
    else
        [F, ~] = stlread(name);
        vertices = F.Points;
        faces = F.ConnectivityList;
    end
end