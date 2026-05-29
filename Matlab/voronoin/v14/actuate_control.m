function chaser = actuate_control(chaser, desired, Sim, Param)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    persistent old_psi_d;

    if isempty(old_psi_d)
        old_psi_d = 0;
    end
    k = Sim.k;

    switch Param.controller
        case 'Mellinger'
           
            pos = chaser.pos(:,k);
            vel = chaser.vel(:,k);
            attitude = reshape(chaser.attitude(:,k),3,3);
            ang_vel = chaser.ang_vel(:,k);

            % get reference
            dpsi_d = (desired.psi - old_psi_d)/Param.dTi;
            old_psi_d = desired.psi;
        
            % mellinger controller
            [T,tau] = drone_mellinger_ctrl(pos,...
                vel, attitude, ang_vel, Param, desired.pos(:,Sim.iD), desired.psi,...
                desired.vel, dpsi_d, desired.acc, desired.jerk);
            
            % nonlinear drone model (continuous time)
            [dot_p,dot_v,dot_R,dot_om] = drone_3dfull_dyn(vel,attitude,ang_vel,T,tau,Param);
            
            % discretization
            chaser.pos(:,k+1) = pos + Param.dTi*dot_p;
            chaser.vel(:,k+1) = vel + Param.dTi*dot_v;
            Rp = rot_integrate(attitude,ang_vel,Param.dTi);
            chaser.attitude(:,k+1) = reshape(Rp,[],1);
            chaser.ang_vel(:,k+1) = ang_vel + Param.dTi*dot_om;
            
        case 'Lee'
        X = struct_to_pack(k, chaser);
        [~, Xtmp] = ode45(@(t,XR) eom(t, XR, desired, Sim, Param), ...
                      [Param.t(k), Param.t(k+1)], ...
                      X);
        chaser = pack_to_struct(k+1, chaser, Xtmp(end,:));
    end
end

