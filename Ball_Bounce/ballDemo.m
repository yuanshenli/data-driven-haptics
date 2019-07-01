%% Ball Bounce Demo 

x1 = 10; %ball position 
x2 = []; %plane position 
for i = 0:length(x2)
    if x1 < x2(i) 
        set_param('ballBounce_dynamic', 'Plane_Input', x2(i));
        set_param('ballBounce_dynamic', 'Ball_Input', x1);
        sim('ballBounce_dynamic');
        x1 = ScopeData1(:,1); %change name to variable name in command window 
    else 
        set_param('ballBounce_dynamic', 'Plane_Input', x2(i));
        set_param('ballBounce_dynamic', 'Ball_Input', x1);
        sim('ballBounce_kinematic');
        x1 = ScopeData1(:,1);
        
    end 
end

%%
subplot(2, 1, 1)
plot(tx, x)
xlabel('time (s)')
ylabel('position (m)')

subplot(2, 1, 2)
plot(tv, v)
xlabel('time (s)')
ylabel('velocity (m/s)')
