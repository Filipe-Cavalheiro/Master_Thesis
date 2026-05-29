function [dp,dv,dR,dom] = drone_3dfull_dyn(v,R,om,T,tau,Param)
%DRONE_MODEL Summary of this function goes here
%   Detailed explanation goes hereParam

    % auxiliary variables
    zW = [0;0;1];
    zB = R(:,3);
    
    % equations of motion
    dp = v;
    dv = -Param.gravity*zW + T/Param.drone_mass*zB - R*Param.drag*R'*v;
    dR = R*skew(om);
    dom = Param.J^(-1)*(-skew(om)*Param.J*om + tau);

%     if t > 0.001
%         test = 1;
%     end
    
end

