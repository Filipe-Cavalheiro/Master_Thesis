function Param = Param_init()    
    % Model and simulation parameters
    Param.Tend = 30;
    Param.dTi = 0.01;  % inner-loop and simulation sampling period
    Param.Nsim = round(Param.Tend/Param.dTi)+1;
    Param.t = 0:Param.dTi:Param.Tend+ Param.dTi;
    Param.gravity = 9.81;     % earth gravity
    Param.nD = 11; % number of drones
    Param.sphere_radius = 2.0;
    Param.nD_drop = 3;
    Param.nD_drop_time = 500;

    % Quadrotor
    Param.drone_mass = 5;
    Param.J = diag([2e-2,2e-2,3e-2]);
    Param.drag = 0.169;
    Param.ctf = 0.0135;

    % measurable_states = 1 -> pos
    % measurable_states = 2 -> pos, vel
    Param.measurable_states = 2;
    
    % options: fibonacci, dijkstra 
    Param.distribuition = 'dijkstra';

    % options: Mellinger, Lee 
    Param.controller = "Mellinger";
    
    % target parameters
    traj = 2;
    switch traj
        case 1
            % Hover
            Param.ref_target_traj.pos = @(t) [ ...
                0; ...
                0; ...
                5 ];
    
            Param.ref_target_traj.vel = @(t) [ ...
                0; ...
                0; ...
                0 ];
    
        case 2
            vel_gain = 0.25;
            % Circular trajectory
            Param.ref_target_traj.pos = @(t) [ ...
                -5 + 15*sin(vel_gain*t); ...
                 15*cos(vel_gain*t); ...
                 5 ];
    
            Param.ref_target_traj.vel = @(t) [ ...
                 vel_gain*15*cos(vel_gain*t); ...
                -vel_gain*15*sin(vel_gain*t); ...
                 0 ];
    
        case 3
            % Lissajous curve
            Param.ref_target_traj.pos = @(t) [ ...
                5*sin(t); ...
                5*sin(2*t); ...
                5 ];
    
            Param.ref_target_traj.vel = @(t) [ ...
                5*cos(t); ...
                10*cos(2*t); ...
                0 ];
    
        case 4
            % 3D Lissajous
            Param.ref_target_traj.pos = @(t) [ ...
                5*sin(t); ...
                5*sin(2*t); ...
                5 + 0.2*cos(2*t) ];
    
            Param.ref_target_traj.vel = @(t) [ ...
                5*cos(t); ...
                10*cos(2*t); ...
                -0.4*sin(2*t) ];
    end
    
    % Initial target position
    Param.initial_target_pos = Param.ref_target_traj.pos(0);
    
    % M690B drone
    % (guessing parameters! needs identification or estimation from geometry)
    % from https://doi.org/10.23919/ACC45564.2020.9147948 we have: "The quadrotor’s mass is M = 1.73Kg and its arms length is L = 0.2m. The moments of inertia matrix used in the nonlinear model (3) is considered diagonal with values: Ixx0=Iyy0=0.03Kg.m2 and Izz0=0.04Kg.m2. "
    % from https://dspace.mit.edu/handle/1721.1/106777 we have as the result of a bifilar pendulum experiment: "mass 𝑚 = 1.28 kg; inertia (𝐽𝑥𝑥, 𝐽𝑦𝑦, 𝐽𝑧𝑧) = (6.9𝐸−3, 7.0𝐸−3, 12.4𝐸−3) Ns2; propeller diameter 0.165 m; propeller positions [±0.117 ±0.117 −0.012] m
    % from https://doi.org/10.1016/j.conengprac.2010.02.008 we have: m = 4.34±5×10−3 kg; IXX = 0.0820±0.0025 kg m2, IYY = 0.0845±0.0029 kg m2, IZZ = 0.1377±0.0059 kg m2; rotor radius r = 0.165 ±0.5×10−3   m; arm length 0.157 m, height 0.035 m.
    % from https://doi.org/10.1109/CEIT.2018.8751873 we have: m = 7.293 kg; Ixx = 0.13, Iyy = 0.25, Izz = 0.55 kg m^2; propeller radius 0.41 or 0.2 m; arm radius 0.55 or 0.45 m.
    Param.Mellinger.kp = diag([100,100,100]);
    Param.Mellinger.kv = diag([15,15,15]);
    Param.Mellinger.kR = diag([8,8,8]);
    Param.Mellinger.kom= diag([0.5,0.5,0.5]);

    % --- Lee Controller gains ---
    Param.Lee.x = 10;
    Param.Lee.v = 8;
    Param.Lee.i = 10;
    Param.c1 = 1.5;
    Param.sigma = 10;
    
    % Attitude
    Param.Lee.R = 1.5;
    Param.Lee.W = 0.35;
    Param.Lee.I = 10;
    Param.c2 = 2;
    
    % Yaw
    Param.Lee.y = 0.8;
    Param.Lee.wy = 0.15;
    Param.Lee.yI = 2;
    Param.c3 = 2;

    % Fixed disturbance
    Param.x_delta = [0.5, 0.8, -1]';
    Param.R_delta = [0.2, 1.0, -0.1]';

    % exponential moving target parameters
    Param.ema_alpha = 0.2;
    Param.noise_std = 0.05; % a 5% error on reading target position
    
    % === Animation ===
    Param.verbose = false;
    Param.animation.animate = true;
    Param.animation.trail_length = 100;
    Param.animation.step = 5;
    % === Create PDF ===
    Param.animation.composed = true;
    Param.animation.radial_hist = true;
    Param.animation.consensus = true;
    Param.animation.create_pdf = true;
    % === Video settings ===
    Param.animation.save_to_mp4 = false;
    Param.animation.frameStep = 10;
    Param.animation.sim_dt = Param.dTi * Param.animation.frameStep;
    Param.animation.videoFPS = 1 / Param.animation.sim_dt;
end