function [Rad, q_d] = update_radius(Rad, q, q_d, c_sphere, Param)
    %UPDATE_RADIUS Summary of this function goes here
    %   Detailed explanation goes here
    % ---------- Radius dynamics (purely scalar) ----------
    q_s  = q   - c_sphere;

    Rad_dot = Param.k_rad * 0.5 * ...
        (1 + q_s' * q_d / (norm(q_s) * norm(q_d))) * ...
        (Param.sphere_radius - Rad);

    Rad = Rad + Param.dTi * Rad_dot;
    
    % update radius of q_d
    q_d = Rad * q_d / norm(q_d);
end

