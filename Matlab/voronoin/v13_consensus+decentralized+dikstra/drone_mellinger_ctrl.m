% Project Capture
% Bruno Guerreiro (bj.guerreiro@fct.unl.pt)
function [thrust,torque,e_p] = drone_mellinger_ctrl(pos,vel,R,om,Param,pos_d,psi_d,ie_p,vel_d,dpsi_d,a_d,jerk_d)

    %     if ~exist('dpsi_d','var') || isempty(dpsi_d), dpsi_ref = 0; end
    %     if ~exist('ie_p','var') || isempty(ie_p), ie_p = zeros(3,1); end
    %     if ~exist('vel_d','var') || isempty(vel_d), vel_d = zeros(3,1); end
    %     if ~exist('a_d','var') || isempty(a_d), a_d = zeros(3,1); end
    %     if ~exist('jerk_d','var') || isempty(jerk_d), jerk_d = zeros(3,1); end

    % auxiliary variables:
    zW = [0;0;1];   % world frame zW is up
    zB = R(:,3);    % drone frame zB is normal to drone plane 

    % define translation errors
    e_p = pos - pos_d;
    e_v = vel - vel_d;
    
    % desired force vector with attitude
    f_d = - Param.Mellinger.kp*e_p - Param.Mellinger.ki*ie_p - Param.Mellinger.kv*e_v + Param.drone_mass*Param.gravity*zW + Param.drone_mass*a_d;

    % compute thrust
    thrust = f_d'*zB; 

    % compute desired rotation matrix
    zB_d = f_d/norm(f_d);                               % agent frame
    xC_d = [cos(psi_d);sin(psi_d);0];                   % world frame
    yB_d = (skew(zB_d)*xC_d)/(norm(skew(zB_d)*xC_d));   % agent frame
    xB_d = skew(yB_d)*zB_d;                             % agent frame
    R_d = [xB_d,yB_d,zB_d];

    % compute desired angular velocity
    hw_d = (Param.drone_mass/thrust)*(jerk_d - (zB_d'*jerk_d)*zB_d);
    ang_p_d = -hw_d'*yB_d;
    ang_q_d = hw_d'*xB_d;
   
    ang_r_d = dpsi_d * (zW' * zB_d); % minimum-snap style yaw rate projection

    om_d = [ang_p_d;
            ang_q_d;
            ang_r_d];

    % compute torques
    e_om = om - om_d;
    e_R = (1/2)*unskew(R_d'*R - R'*R_d);
    torque = -Param.Mellinger.kR*e_R - Param.Mellinger.kom*e_om;
    
%     if t > 0.001
%         fprintf('t = %f: thrust = %f; torque = [%f %f %f]; e_p = [%f %f %f]; e_v = [%f %f %f]; f_d = [%f %f %f]; e_R = [%f %f %f]; e_om = [%f %f %f].\n',...
%             t,thrust,torque(1),torque(2),torque(3),e_p(1),e_p(2),e_p(3),e_v(1),e_v(2),e_v(3),...
%             f_d(1),f_d(2),f_d(3),e_R(1),e_R(2),e_R(3),e_om(1),e_om(2),e_om(3));
%         test = 1;
%     end
    
end

