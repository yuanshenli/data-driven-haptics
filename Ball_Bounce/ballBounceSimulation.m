clc;
clear;
close all;

%% simulation params
dt = 0.0002;          % time resolution [s]
T = 0 : dt : 2;
g = -9.81;          % [m/s2]

%% ball params
m1 = 0.1;    % [kg]
% Initial conditions
x1_0 = 0.1;  % [m]
v1_0 = 0;    % [m/s]
a1_0 = g;    % [m/s2]
% History
x1 = zeros(length(T), 1);    % [m]
v1 = zeros(length(T), 1);    % [m/s]
a1 = zeros(length(T), 1);    % [m/s2]
F1 = zeros(length(T), 1);    % contact force on the ball [N]

%% plane parmas
m2 = 50;    % [kg]
% Initial conditions
x2_0 = 0;   % [m]
v2_0 = 0;   % [m/s]
a2_0 = 0;   % [m/s2]
% History
x2 = zeros(length(T), 1);    % [m]
v2 = zeros(length(T), 1);    % [m/s]
a2 = zeros(length(T), 1);    % [m/s2]
F2 = zeros(length(T), 1);    % contact force on the hand [N]

%% Dynamic params
k = 100.5;   % contact stiffness [N/m]
b = 4;    % damping [Ns/m]

%% simulation
% Init
x1(1) = x1_0;
v1(1) = v1_0;
a1(1) = a1_0;

x2(1) = x2_0;
v2(1) = v2_0;
a2(1) = a2_0;

for ii = 1:length(T)
    
    if (ii > 1)
        x1(ii) = x1(ii-1) + v1(ii-1) * dt + a1(ii-1)/2 * dt^2;
        v1(ii) = v1(ii-1) + a1(ii-1) * dt;
    end
    
    if (x1(ii) > x2(ii))
        F1(ii) = 0;
    else 
        F1(ii) = k * (x2(ii) - x1(ii)) + b * (v2(ii) - v1(ii)) - m1 * g;
        F1(ii) = max(0, F1(ii));
%         F1(ii) = k * (x2(ii) - x1(ii));
    end
    
    a1(ii) = F1(ii) / m1 + g;
    
    

    
end

%% plot
figure(1);
subplot(2, 2, 1);
plot(T, x1);
xlabel("Time [s]");
ylabel("height [m]");

subplot(2, 2, 2);
plot(T, v1);
xlabel("Time [s]");
ylabel("velocity [m/s]");

subplot(2, 2, 3);
plot(T, a1);
xlabel("Time [s]");
ylabel("acceleration [m/s^{2}]");

subplot(2, 2, 4);
plot(T, F1);
xlabel("Time [s]");
ylabel("force [N]");

figure(2);

plot(T, x1, 'r');
hold on;
plot(T, v1, 'g');
hold on;
plot(T, a1, 'b');
hold on;
plot(T, F1, 'k');
xlabel("Time [s]");
legend(["height [m]", "velocity [m/s]", "acceleration [m/s^{2}]", "force [N]"])


