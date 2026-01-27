function [Rad,q, q_d] = linear_motion_ref(Rad,q,omega,q_d,omega_d,Param)
    % System parameters (V. Numerical Simulations)
    % original values: 
    % c = 0.3;                             % Nm (s/rad)
    % J = diag([0.0294 0.0305 0.0495]);    % kg m^2
    % tau_outerloop = zeros(3,1);

    % current values:
    c = 0.3;                             % Nm (s/rad)
    J = diag([0.00294 0.00305 0.00495]);     % kg m^2
    tau_outerloop = zeros(3,1);
    
    % Controller gains (29a, 29b)
    K_r = diag([40 40 70]);
    K_w = diag([7 7 12]) * 1e-1;
    
    % skew operator
    S = @(x) [   0   -x(3)  x(2);
               x(3)    0   -x(1);
              -x(2)  x(1)    0 ];
    
    Q = eye(3);

    % --- alternative computations
    Rad_dot = Param.k_rad*0.5*(1+q'*q_d/(norm(q)*norm(q_d)))*(Param.sphere_radius - norm(q));
    Rad = Rad + Param.dTi * Rad_dot;
    
    %pass points to unit sphere
    q_hat   = projectToSphere(q, 1);
    qd_hat  = projectToSphere(q_d, 1);

    % --- Configuration error vector (Eq. 12)
    % this equation requires q and q_d to be in the unit sphere
    e_q = (1 / (sqrt(2) * sqrt(1 + (q_hat' * qd_hat)))) * Q' * S(qd_hat) * q_hat;
    
    % --- Angular velocity error
    e_w = omega - omega_d;

    % --- Control law (Eq. 28, reduced to S^2)
    u = Q'* (-K_r * Q * e_q ...
        -K_w * e_w )...
        + S(omega) * J * omega ...
        + c * omega ...
        - tau_outerloop;

    % --- Rigid body dynamics (Eq. 5)
    omega_dot = J \ (u - c*omega - S(omega)*J*omega + tau_outerloop);
    omega = omega + Param.dTi * omega_dot;

    % --- Reduced attitude kinematics (Eq. 4)
    q_dot = S(omega) * q;
    q = q + Param.dTi * q_dot;

    % --- Enforce S^2 constraint
    q = (q / norm(q))*Rad;
    q_d = (q_d / norm(q_d))*Rad;
    
    % --- Rotate q_d along S^2
    % q_d_dot = S(omega_d) * q_d;
    % q_d = q_d + Param.dTi * q_d_dot;