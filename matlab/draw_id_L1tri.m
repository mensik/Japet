function draw_id_L1tri(x,e,u,l)

% draw_id_L1tri(x,e,u)
%          displays a vector curl field

ne = size(e,2);

%figure
hold on
for i=1:ne
    x1 = x(:,e(1,i));
    x2 = x(:,e(2,i));
    x3 = x(:,e(3,i));
    
    u1 = u(e(1,i));
    u2 = u(e(2,i));
    u3 = u(e(3,i));
    
    plot3([x1(1),x2(1),x3(1),x1(1)],[x1(2),x2(2),x3(2),x1(2)],[u1,u2,u3,u1]);
end

nL = size(l,1);
for i=1:nL;
    if l(i) > 1e-5
       plot3([x(1,i) x(1,i)],[x(2,i) x(2,i)],[u(i) u(i)-l(i)],...
            'LineWidth',2,...
            'Color','red',...
            'Marker','.');
       
    end
end

%hold off
