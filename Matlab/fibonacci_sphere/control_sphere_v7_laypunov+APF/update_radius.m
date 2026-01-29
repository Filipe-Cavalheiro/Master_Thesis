function [Rad, q_d] = update_radius(Rad, q, q_d, c_sphere, Param)
    %UPDATE_RADIUS Summary of this function goes here
    %   Detailed explanation goes here
    % ---------- Radius dynamics (purely scalar) ----------

    Rad_dot = Param.k_rad * 0.5 * ...
        (1 + q' * q_d / (norm(q) * norm(q_d))) * ...
        (Param.sphere_radius - Rad);

    Rad = Rad + Param.dTi * Rad_dot;
    
    % update radius of q_d
    q_d = projectToSphere(q_d, Rad, c_sphere);
end
