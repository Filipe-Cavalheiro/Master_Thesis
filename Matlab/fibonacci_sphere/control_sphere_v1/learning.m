close all;
clearvars;

% Model data
kfinal = 60;
A = eye(3);
B = eye(3);
C = eye(3);
nx = size(B, 1);
nu = size(B, 2);
ny = size(C, 1);

% MPC data
N = 1; 
P = eye(nx);
Q = eye(nx);
R = eye(nu);

x0 = [-4; -3; 2];      % initial condition
ref = @(k)zeros(nx, 1); % reference signal

F = eye(nx);
G = zeros(nx, N*nu);
Gaux = [];
for n = 1:N
    F = [F; A^n];
    Gaux = [A^(n-1)*B Gaux];
    G = [G; Gaux zeros(nx, (N-n)*nu)];
end

H = kron(eye(N+1), C);
F_bar = H*F;
G_bar = H*G;
Q_bar = blkdiag(kron(eye(N), Q), P);
R_bar = kron(eye(N), R);

R_til = G_bar'*Q_bar*G_bar+R_bar;
S_til = G_bar'*Q_bar;

Ky = R_til^(-1)*S_til;
K = Ky*F_bar;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%simulation initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

states = [];
input = [];
reference = [];
for k = 0:kfinal
    pred_states{k+1} = [];
    pred_inputs{k+1} = [];
end

x = x0;
Y_bar = [];
for k = 0:N
    Y_bar = [Y_bar; ref(k)];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%simulation loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for k = 0:kfinal
    U = -K*x + Ky*Y_bar;
    X = F*x + G*U;
    u = U(1:nu);

    states = [states x];
    input = [input u];
    reference = [reference ref(k)];
    pred_states{k+1} = reshape(X, nx, N+1);

    % Next State and Reference Update
    x = A*x + B*u;
    Y_bar = [Y_bar(ny+1:end); ref(k+N+1)];
end

% Drawing the trajectory of the system 
figure;
axis([-5 5 -5 5 -5 5])
hold on;
% plannedPath = plot(xtraj(1,1),xtraj(2,1),'Color',[0 0 0.5020],'LineStyle',':','HandleVisibility','off');
dronePath = animatedline('Color',[220 220 220]/255,'HandleVisibility','off');
drones = scatter3(states(1,1),states(2,1), states(3,1),'LineWidth',0.6,'MarkerEdgeColor','k','MarkerFaceColor','k','DisplayName','drone position');
view(3);
grid on;
for k = 1:(kfinal+1)
    addpoints(dronePath,states(1,k),states(2,k), states(3,k));
    set(drones,'XData',states(1,k),'YData',states(2,k), 'ZData', states(3,k));
    drawnow
    pause(0.01)
end