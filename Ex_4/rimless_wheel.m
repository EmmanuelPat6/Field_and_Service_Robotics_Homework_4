clc
close all
clear all

% Parameters
g = 9.81;          % gravity (m/s^2)
l = 1;           % leg length (m) %%MODIFY HERE%% 
alpha = pi/8;      % half inter-leg angle (rad) %%MODIFY HERE%% 
gamma = 0.08;      % slope angle (rad) %%MODIFY HERE%% 

% Initial conditions
thetadot0 = 0.95;%%MODIFY HERE%% 
if (thetadot0 >= 0)
    theta0 = gamma-alpha;
else
    theta0 = gamma+alpha;
end

double_support = 0;

y0 = [theta0; thetadot0];

% Simulation settings
t0 = 0; %initial time
tf = 25; %final time
dt = 0.01; %max step time

% Time/state storage
T = [];
Y = [];

while t0 < tf
    options = odeset('Events', @(t, y) impact_event(t, y, alpha,gamma), 'MaxStep', dt);
    [t, y, te, ye, ie] = ode45(@(t, y) dynamics(t, y, g, l, double_support), [t0 tf], y0, options);
    
    T = [T; t];
    Y = [Y; y];

    if ~isempty(te)
        [y0,double_support] = impact_map(ye, alpha,g,l); % apply impact map
        t0 = te;
    else
        break;
    end
end

% Plot results
figure(1)
plot(T, Y(:,1), 'b', 'DisplayName', '\theta (rad)');
hold on;
plot(T, Y(:,2), 'r', 'DisplayName', '\theta dot (rad/s)');
xlabel('Time (s)');
ylabel('State');
title('Rimless Wheel Dynamics');
legend show;
grid on;

figure(2)
plot(Y(:,1), Y(:,2), 'b', 'DisplayName', '\theta (rad)');
hold on
plot(Y(1,1), Y(1,2), 'r', 'Marker','*','LineWidth',5,'DisplayName','Initial point');
xlabel('\theta (rad)');
ylabel('\theta dot (rad/s)');
title('Rimless Wheel Limit Cycle');
legend show;
grid on;


% Save PDF
figureNumbers = [1, 2];
fileNames = {'rimless_wheel_dynamics', 'rimless_wheel_limit_cycle'};

for i = 1:length(figureNumbers)
    fig = figure(figureNumbers(i));
    
    % Imposta dimensione precisa della figura (es. 6x4 pollici)
    set(fig, 'Units', 'inches', 'Position', [1 1 6 4]);

    % Imposta formato PDF senza bordi
    set(fig, 'PaperUnits', 'inches');
    set(fig, 'PaperSize', [6 4]);
    set(fig, 'PaperPosition', [0 0 6 4]);

    % Salva in PDF
    print(fig, fileNames{i}, '-dpdf');
end

function dydt = dynamics(~, y, g, l, ds)
    theta = y(1);
    thetadot = y(2);
    if (~ds)
        dtheta = thetadot;
        dthetadot = (g/l) * sin(theta);
    else
        dtheta = 0;
        dthetadot = 0;
    end
    dydt = [dtheta; dthetadot];
end


function [value, isterminal, direction] = impact_event(~, y, alpha,gamma)
    
    value = [y(1)-alpha-gamma; y(1)-gamma+alpha];% Trigger when theta = gamma+alpha
                                     %Trigger when theta = gamma-alpha
    isterminal = [1;1];         % Stop the integration
    direction = [1;-1];          % Detect only when increasing
end

function [yplus,ds] = impact_map(y_minus, alpha,g,l)%minus: before impact time; plus: after impact time
    if (y_minus(2)>=0)
        theta_plus = y_minus(1)-2*alpha;
    else
        theta_plus = y_minus(1)+2*alpha;
    end
    thetadot_plus = cos(2*alpha) * y_minus(2);
    if (thetadot_plus < 0.01*sqrt(g/l) && thetadot_plus >-0.01*sqrt(g/l)) 
        thetadot_plus = 0;
        ds = 1;
    else
        ds = 0;
    end
    yplus = [theta_plus; thetadot_plus];
end