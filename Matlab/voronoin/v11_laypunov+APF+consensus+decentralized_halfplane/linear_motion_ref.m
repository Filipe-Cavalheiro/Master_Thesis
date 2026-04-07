function [q, e_q] = linear_motion_ref( ...
            Rad, q, omega, q_d, omega_d, c_sphere, Param)

    % System parameters (V. Numerical Simulations)
    % original values:
    % c = 0.3;                             % Nm (s/rad)
    % J = diag([0.0294 0.0305 0.0495]);    % kg m^2
    % tau = zeros(3,1);
    % 
    % % Controller gains (29a, 29b)
    % K_r = diag([4.234 4.392 7.128]);
    % K_w = diag([7.056 7.320 11.88]) * 1e-1;

    Q = eye(3);

    % ---------- Shift to sphere frame (back to origin)----------
    q_s  = q   - c_sphere;
    q_d_s  = q_d   - c_sphere;

    % ---------- Unit-sphere projection ----------
    q_hat  = q_s/ norm(q_s);
    qd_hat = q_d_s / norm(q_d_s);
    
    % ---------- Configuration error ----------
    e_q = (1 / (sqrt(2) * sqrt(1 + q_hat' * qd_hat))) * ...
          Q' * skew(qd_hat) * q_hat;

    % ---------- Angular velocity error ----------
    e_w = omega - omega_d;

    % ---------- Control law ----------
    u = Q' * (-Param.K_r * Q * e_q - Param.K_w * e_w) ...
        + skew(omega) * Param.J * omega ...
        + Param.c * omega ...
        - Param.tau;

    % ---------- Angular dynamics ----------
    omega_dot = Param.J \ (u - Param.c*omega - skew(omega)*Param.J*omega + Param.tau);
    omega = omega + Param.dTi * omega_dot;

    % ---------- Reduced attitude kinematics ----------
    q_dot = skew(omega) * q_s;
    q_s = q_s + Param.dTi * q_dot;

    % ---------- Enforce S² constraint (local) ----------
    q_s = Rad * q_s  / norm(q_s);

    % ---------- Shift back to world frame ----------
    q = q_s  + c_sphere;
end
