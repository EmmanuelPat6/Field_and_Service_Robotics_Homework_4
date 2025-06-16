% Representation-Free Model Predictive Control for Dynamic Quadruped Panther
% Author: Yanran Ding
% Last modified: 2020/12/21
% 
% Code accompanying the paper:
% Yanran Ding, Abhishek Pandala, Chuanzheng Li, Young-Ha Shin, Hae-Won Park
% "Representation-Free Model Predictive Control for Dynamic Motions in Quadrupeds"
% Transactions on Robotics
% 
% preprint available at: https://arxiv.org/abs/2012.10002
% video available at: https://www.youtube.com/watch?v=iMacEwQisoQ&t=101s

%% initialization
clear all;close all;clc
addpath fcns fcns_MPC

%% --- parameters ---
% ---- gait ----
% 0-trot; 1-bound; 2-pacing 3-gallop; 4-trot run; 5-crawl
gait = 0; 
p = get_params(gait);
p.playSpeed =1;
p.flag_movie = 1;       % 1 - make movie

dt_sim = p.simTimeStep;
SimTimeDuration = 4.5;  % [sec]
MAX_ITER = floor(SimTimeDuration/p.simTimeStep);

% desired trajectory
p.acc_d = 1;
p.vel_d = [0.5;0];
p.yaw_d = 0;

%% Model Predictive Control
% --- initial condition ---
% Xt = [pc dpc vR wb pf]': [30,1]
if gait == 1
    [p,Xt,Ut] = fcn_bound_ref_traj(p);
else
    [Xt,Ut] = fcn_gen_XdUd(0,[],[1;1;1;1],p);
end

% --- logging ---
tstart = 0;
tend = dt_sim;

[tout,Xout,Uout,Xdout,Udout,Uext,FSMout] = deal([]);

% --- simulation ----
h_waitbar = waitbar(0,'Calculating...');
tic
for ii = 1:MAX_ITER
    % --- time vector ---
    t_ = dt_sim * (ii-1) + p.Tmpc * (0:p.predHorizon-1);
    
    % --- FSM ---
    if gait == 1

        [FSM,Xd,Ud,Xt] = fcn_FSM_bound(t_,Xt,p);
    else
        [FSM,Xd,Ud,Xt] = fcn_FSM(t_,Xt,p);
    end

    % --- MPC ----
    % form QP
    [H,g,Aineq,bineq,Aeq,beq] = fcn_get_QP_form_eta(Xt,Ut,Xd,Ud,p);

    %%
    % Considering the matrices for the QP obtained from function fcn_get_QP_form_eta, use the QP solver qpSWIFT to 
    %  solve the quadratic problem with the following form 
    %  min. 0.5 * x' * H *x + g' * x
    %  s.t. Aineq *x <= bineq
    %      Aeq * x = beq
    % 
    % The result of the QP problem should be stored in a variable called zval in order to be used in the following
    
    [zval,basic_info] = qpSWIFT(sparse(H),g,sparse(Aeq),beq,sparse(Aineq),bineq);
    
    %%
    
    
    Ut = Ut + zval(1:12);
    
    % --- external disturbance ---
    [u_ext,p_ext] = fcn_get_disturbance(tstart,p);
    p.p_ext = p_ext;        % position of external force
    u_ext = 0*u_ext;
    
    % --- simulate ---
    [t,X] = ode45(@(t,X)dynamics_SRB(t,X,Ut,Xd,0*u_ext,p),[tstart,tend],Xt);
    
    
    % --- update ---
    Xt = X(end,:)';
    tstart = tend;
    tend = tstart + dt_sim;
    
    % --- log ---  
    lent = length(t(2:end));
    tout = [tout;t(2:end)];
    Xout = [Xout;X(2:end,:)];
    Uout = [Uout;repmat(Ut',[lent,1])];
    Xdout = [Xdout;repmat(Xd(:,1)',[lent,1])];
    Udout = [Udout;repmat(Ud(:,1)',[lent,1])];
    Uext = [Uext;repmat(u_ext',[lent,1])];
    FSMout = [FSMout;repmat(FSM',[lent,1])];
    
    waitbar(ii/MAX_ITER,h_waitbar,'Calculating...');
end
close(h_waitbar)
fprintf('Calculation Complete!\n')
toc

%% Animation
[t,EA,EAd] = fig_animate(tout,Xout,Uout,Xdout,Udout,Uext,p);

%% General Plots
fig = figure(1);
set(fig, 'Units', 'inches');
figPosition = get(fig, 'Position');  % [left bottom width height]

set(fig, 'PaperUnits', 'inches');
set(fig, 'PaperSize', figPosition(3:4));
set(fig, 'PaperPosition', [0 0 figPosition(3:4)]); % No margin

print(fig, 'general_plot', '-dpdf')


%% ALL FINAL PLOTS
fig_all = figure('Name', 'All_Plots', 'Position', [100 100 1800 900]);
t = tiledlayout(2,2, 'Padding', 'compact', 'TileSpacing', 'compact');

%% Fz
nexttile
plot(tout,Uout(:,3),'r', 'LineWidth', 2); hold on
plot(tout,Uout(:,6),'g', 'LineWidth', 2)
plot(tout,Uout(:,9),'b', 'LineWidth', 2)
plot(tout,Uout(:,12),'k', 'LineWidth', 2)
plot(tout,Udout(:,3),'r--', 'LineWidth', 1.5)
plot(tout,Udout(:,6),'g--', 'LineWidth', 1.5)
plot(tout,Udout(:,9),'b--', 'LineWidth', 1.5)
plot(tout,Udout(:,12),'k--', 'LineWidth', 1.5)
xlabel('$t [s]$', 'Interpreter', 'latex', 'FontSize', 18)
ylabel('$F_z [N]$', 'Interpreter', 'latex', 'FontSize', 18)
title('Fz', 'FontSize', 20)
grid on; box on; xlim([0, tout(end)])
legend({'$F_{z_1}(t)$','$F_{z_2}(t)$','$F_{z_3}(t)$','$F_{z_4}(t)$'}, ...
    'Interpreter', 'latex', 'FontSize', 16, 'Location', 'best');

%% Position
nexttile
plot(tout,Xout(:,1),'r', 'LineWidth', 2); hold on
plot(tout,Xout(:,2),'g', 'LineWidth', 2)
plot(tout,Xout(:,3),'b', 'LineWidth', 2)
plot(tout,Xdout(:,1),'r--', 'LineWidth', 1.5)
plot(tout,Xdout(:,2),'g--', 'LineWidth', 1.5)
plot(tout,Xdout(:,3),'b--', 'LineWidth', 1.5)
xlabel('$t [s]$', 'Interpreter', 'latex', 'FontSize', 18)
ylabel('$p [m]$', 'Interpreter', 'latex', 'FontSize', 18)
title('Position', 'FontSize', 20)
grid on; box on; xlim([0, tout(end)])
legend({'$x(t)$','$y(t)$','$z(t)$'}, 'Interpreter', 'latex', 'FontSize', 16, 'Location', 'best');

%% Velocity
nexttile
plot(tout,Xout(:,4),'r', 'LineWidth', 2); hold on
plot(tout,Xout(:,5),'g', 'LineWidth', 2)
plot(tout,Xout(:,6),'b', 'LineWidth', 2)
plot(tout,Xdout(:,4),'r--', 'LineWidth', 1.5)
plot(tout,Xdout(:,5),'g--', 'LineWidth', 1.5)
plot(tout,Xdout(:,6),'b--', 'LineWidth', 1.5)
xlabel('$t [s]$', 'Interpreter', 'latex', 'FontSize', 18)
ylabel('$v [m/s]$', 'Interpreter', 'latex', 'FontSize', 18)
title('Velocity', 'FontSize', 20)
grid on; box on; xlim([0, tout(end)])
legend({'$v_x(t)$','$v_y(t)$','$v_z(t)$'}, 'Interpreter', 'latex', 'FontSize', 16, 'Location', 'best');

%% Angular Velocity
nexttile
plot(tout,Xout(:,16),'r', 'LineWidth', 2); hold on
plot(tout,Xout(:,17),'g', 'LineWidth', 2)
plot(tout,Xout(:,18),'b', 'LineWidth', 2)
plot(tout,Xdout(:,16),'r--', 'LineWidth', 1.5)
plot(tout,Xdout(:,17),'g--', 'LineWidth', 1.5)
plot(tout,Xdout(:,18),'b--', 'LineWidth', 1.5)
xlabel('$t [s]$', 'Interpreter', 'latex', 'FontSize', 18)
ylabel('$\omega [rad/s]$', 'Interpreter', 'latex', 'FontSize', 18)
title('Angular Velocity', 'FontSize', 20)
grid on; box on; xlim([0, tout(end)])
legend({'$\omega_x(t)$','$\omega_y(t)$','$\omega_z(t)$'}, ...
    'Interpreter', 'latex', 'FontSize', 16, 'Location', 'best');

% PDF
exportgraphics(fig_all, 'All_Plots.pdf', 'ContentType', 'vector');



