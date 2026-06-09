function [f_safe, pos_cmd] = R3_spherical_CBF(f_nom, X, Sim, Param)
[pos, vel, R, W, ei, ~] = split_to_states(X);
safe_center = [0;0;0];
h = 1-((pos-safe_center)/1).^4;
Lfh = 
end

