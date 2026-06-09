function [delta, om_b] = att_controller(v_air, beta, alpha, lbd_b, x, P)

    V = norm(v_air);
    pd = (P.rho*V^2)/2;

    R = reshape(x(1:9,1),3,3);
    lbd = R2Euler(R);
    om = x(10:12,1);

    gamma = lbd(2) - alpha*cos(lbd(1)) - beta*sin(lbd(1));
    
    e_phi = lbd(1) - lbd_b(1);
    e_theta = lbd(2)-lbd_b(2);
    e_psi = lbd(3) - lbd_b(3);
    
    p_b = -P.mu_phi*e_phi - tan(lbd(2))*sin(lbd(1))*om(2) - tan(lbd(2))*cos(lbd(1))*om(3);
    q_b = (-P.mu_theta*e_theta + om(3)*sin(lbd(1)))/cos(lbd(1));
    r_b = -P.mu_psi*e_psi*(cos(lbd(2))/cos(lbd(1))) - (sin(lbd(1))/cos(lbd(1)))*om(2);
    om_b = [p_b;q_b;r_b];
    e_om = om - om_b;
    om_b_d = zeros(3,1);

    A1 = (P.I(1,1)/(pd*P.S*P.b))*(-e_phi - P.mu_p*e_om(1) + ((P.I(3,3)-P.I(2,2))/P.I(1,1))*om(2)*om(3) - (P.I(1,3)/P.I(1,1))*om(1)*om(2) + om_b_d(1)) - (P.Clb*beta + (P.b/(2*V))*(P.Clp*om(1)+P.Clr*om(3)));
    A2 = (P.I(3,3)/(pd*P.S*P.b))*(-e_psi - P.mu_r*e_om(3) + ((P.I(2,2)-P.I(1,1))/P.I(3,3))*om(1)*om(2) + (P.I(1,3)/P.I(3,3))*om(2)*om(3) + om_b_d(3)) - (P.Cnb*beta + (P.b/(2*V))*(P.Cnp*om(1)+P.Cnr*om(3)));
    
    delta_r =  (A1*P.Cnda-A2*P.Clda)/(P.Cldr*P.Cnda-P.Clda*P.Cndr);
    delta_a = (-A1*P.Cndr+A2*P.Cldr)/(P.Cldr*P.Cnda-P.Clda*P.Cndr);
    
    delta_1 = -e_theta*cos(lbd(1)) - P.mu_q*e_om(2);
    delta_2 = P.Cm0 + P.Cma*((lbd(2)-gamma-beta*sin(lbd(1)))/cos(lbd(1))) + (P.c_/2*V)*P.Cmq*om(2);
    delta_3 = ((P.I(1,1) - P.I(3,3))/P.I(2,2))*om(1)*om(3) + (P.I(1,3)/P.I(2,2))*(om(1)^2-om(3)^2) + om_b_d(2);
    
    delta_e = (P.I(2,2)/(pd*P.S*P.c_*P.Cmde))*(delta_1 - ((pd*P.S*P.c_)/P.I(2,2))*delta_2 + delta_3);

    delta = [delta_e;delta_r;delta_a];

end
