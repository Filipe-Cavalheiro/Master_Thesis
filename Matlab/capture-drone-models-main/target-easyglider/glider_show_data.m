% Project Capture
% Bruno Guerreiro (bj.guerreiro@fct.unl.pt)
% 
% Summary: script that plots basic simulation results

imgs_folder = 'figures/';
saves_folder = 'saves/';
if ~exist('filename','var') || isempty(filename)
    filename = [datestr(now,30) '_simul_glider'];
end
do_print = 0; % set to 1 to plot each plot in a PDF file (for latex reports)
do_save_workspace = 0; % set to 1 to save workspace to *.mat file

% get data from simulink
t = out.simout.time;
data = out.simout.signals.values;

[n1,n2] = size(out.simout.signals.values);
if n1 > n2 % diferent Matlab versions use the transpose of the data values
    data = data';
end
v_air = data((1:3),:);
lbd_b = data(3+(1:3),:);
x = data(3+3+(1:12),:);
u = data(3+3+12+(1:3),:);
lbd = data(3+3+12+3+(1:3),:);

% show results plot
set(0,'defaultTextInterpreter','latex');
set(0,'defaultLegendInterpreter','latex');
sstblue         = [0,128,255]/255;
sstlightblue    = [48,208,216]/255;
sstlighterblue  = [50,220,240]/255;
sstlightestblue = [60,230,255]/255;
sstgreen        = [43,191,92]/255;
sstlightgreen   = [140,255,200]/255;
sstlightergreen = [160,255,225]/255;
sstgray         = [70,70,70]/255;
sstlightgray    = [200,200,200]/255;

dcolors = { sstgreen, sstblue, sstlightblue, sstlighterblue, sstlightestblue, sstlightgreen, sstlightergreen, sstlightgray };

sstgray = [70,70,70]/255;
nt = length(t);
dt = mean(t(2:end)-t(1:end-1));

figure(101);
subplot(411);
plot(t,vecnorm(v_air),'Color',dcolors{1});
grid on;
ylabel('$$V_{air}$$ [m/s]');
title('Control variables');
subplot(412);
plot(t,u(1,:),'Color',dcolors{1});
grid on;
ylabel('$$\delta_e(t)$$ [rad]');
subplot(413);
plot(t,u(2,:),'Color',dcolors{1});
grid on;
ylabel('$$\delta_r(t)$$ [rad]');
subplot(414);
plot(t,u(3,:),'Color',dcolors{1});
grid on;
ylabel('$$\delta_a(t)$$ [rad]');
print2pdf([imgs_folder filename '_act'],do_print);

figure(104);
subplot(311);
plot(t,lbd_b(1,:)*180/pi,'Color',sstgray);
hold on;
plot(t,lbd(1,:)*180/pi,'Color',dcolors{1});
hold off;
grid on;
ylabel('$$\phi(t)$$ [deg]');
title('Attitude (euler angles)');
subplot(312);
plot(t,lbd_b(2,:)*180/pi,'Color',sstgray);
hold on;
plot(t,lbd(2,:)*180/pi,'Color',dcolors{1});
hold off;
grid on;
ylabel('$$\theta(t)$$ [deg]');
subplot(313);
plot(t,lbd_b(3,:)*180/pi,'Color',sstgray);
hold on;
plot(t,lbd(3,:)*180/pi,'Color',dcolors{1});
hold off;
grid on;
xlabel('$$t$$ [s]');
ylabel('$$\psi(t)$$ [deg]');
print2pdf([imgs_folder filename '_eul'],do_print);

figure(105);
subplot(311);
plot(t,x(10,:),'Color',dcolors{1});
grid on;
ylabel('$$\omega_x(t)$$ [rad]');
title('Attitude (euler angles)');
subplot(312);
plot(t,x(11,:),'Color',dcolors{1});
grid on;
ylabel('$$\omega_y(t)$$ [rad]');
subplot(313);
plot(t,x(12,:),'Color',dcolors{1});
grid on;
xlabel('$$t$$ [s]');
ylabel('$$\omega_z(t)$$ [rad]');
print2pdf([imgs_folder filename '_om'],do_print);

if do_save_workspace
    save([saves_folder filename '.mat']);
end

