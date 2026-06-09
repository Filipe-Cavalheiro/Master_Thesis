function keyHandler(src, event)
    data = guidata(src);
    state = data.state;
    gfx   = data.gfx;
    Param = data.Param;

    switch event.Key
        case 'uparrow'
            % same object, new random sites
            [state.vertices, state.faces] = loadObject(state, Param);
            N = size(state.vertices,1);
            state.sites = randperm(N, Param.nD);
            visualizeState(state, 'New random sites', gfx, Param);
            data.sites = state.sites;
            guidata(src, data);

        case 'rightarrow'
            % next object
            state.obj_index = state.obj_index + 1;
            if state.obj_index > numel(Param.objList)
                state.obj_index = 1;
            end
            [state.vertices, state.faces] = loadObject(state, Param);
            state.sites = randperm(size(state.vertices,1), Param.nD);
            visualizeState(state, 'Next object', gfx, Param);
            data.sites = state.sites;
            guidata(src, data);

        case 'leftarrow'
            % previous object
            state.obj_index = state.obj_index - 1;
            if state.obj_index < 1
                state.obj_index = numel(Param.objList);
            end
            [state.vertices, state.faces] = loadObject(state, Param);
            state.sites = randperm(size(state.vertices,1), Param.nD);
            visualizeState(state, 'Previous object', gfx, Param);
            data.sites = state.sites;
            guidata(src, data);

        case 'return'
            % run algorithm
            % for iter = 1:Param.maxIter
            %     newSites = runCVT(state, gfx);
            %     if isequal(sort(newSites), sort(state.sites))
            %         fprintf('Converged at iteration %d\n', iter);
            %         break;
            %     end
            %     state.sites = newSites;
            % end
            data.state = state;
            guidata(src, data);
            uiresume(src)
            return;
            
        otherwise
            return;
    end

    data.state = state;
    guidata(src, data);
end