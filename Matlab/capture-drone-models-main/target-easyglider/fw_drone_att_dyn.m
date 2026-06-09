function [dR,dom] = fw_drone_att_dyn(v_air,R,om,delta,P)

    % auxiliary variables:
    V = norm(v_air);
    pd = (P.rho*V^2)/2;
    alpha = P.alpha;
    beta = P.beta;

    % Compute aerodynamics coeficientsP = Param;
    de = delta(1);
    dr = delta(2);
    da = delta(3);    
    Cl = P.Clb*beta + P.Cldr*dr + P.Clda*da + 0.7*(P.b/V)*(P.Clp*om(1) + P.Clr*om(3));
    Cm = P.Cm0 + P.Cma*alpha + P.Cmde*de + (P.c_/2*V)* P.Cmq*om(2);
    Cn = P.Cnb*P.beta + P.Cndr*dr + P.Cnda*da + 0.7*(P.b/V)*(P.Cnp*om(1) + P.Cnr*om(3));

    % Compute angular dynamics
    C1 = (((pd*P.S*P.b )/P.I(1,1))*Cl) - (((P.I(3,3)-P.I(2,2))/P.I(1,1))*om(2)*om(3)) + ((P.I(1,3))/P.I(1,1))*om(1)*om(2);
    C2 = (((pd*P.S*P.c_)/P.I(2,2))*Cm) - (((P.I(1,1)-P.I(3,3))/P.I(2,2))*om(1)*om(3)) - ((P.I(1,3)/P.I(2,2))*(om(1)^2 - om(3)^2)) + (P.Ie/P.I(2,2))*P.omega*om(3);
    C3 = (((pd*P.S*P.b )/P.I(3,3))*Cn) - (((P.I(2,2)-P.I(1,1))/P.I(3,3))*om(1)*om(2)) - ((P.I(1,3)/P.I(3,3))*om(2)*om(3)) - ((P.Ie/P.I(3,3))*P.omega*om(2));
    
    p_dot = (P.I(3,3)/(P.I(1,1)*P.I(3,3)-P.I(1,3)^2)) * (C1*P.I(1,1) + C3*P.I(1,3));
    q_dot = C2;
    r_dot = (P.I(1,1)/(P.I(1,1)*P.I(3,3)-P.I(1,3)^2)) * (C1*P.I(1,3) + C3*P.I(3,3));
    dom=[p_dot;q_dot;r_dot];

%     tauA = 1/2*P.rho*V^2*P.S*[ P.b*Cl; P.c_*Cm; P.b*Cn];
%     tauP = skew(om)*[P.Ie*P.omega;0;0];
%     dom2 = P.I^(-1)*(-skew(om)*P.I*om + tauP + tauA);
%     comp = [dom,dom2]

    % angular kinematics:
    dR = R*skew(om);

end