function [x,e] = discretRectangle(k,l,m,n,h, pi)
% Diskretizuje obdelnik
% [x,e,IU,ID,IL,IR] = discretRectangle(k,l,m,n,h,pi)
% Diskretizuje obdelnik <k,l> x <m,n> s krokem h. Pocetecni index pi 

x1 = k:h:l;
x2 = m:h:n;

nx1 = max(size(x1));
nx2 = max(size(x2));

x = [];
e = [];

index = pi;

% seznam uzlu a elementu
for j=1:nx2
    for i=1:nx1
        x = [x [x1(i);x2(j)]];
        
        if i<nx1 && j<nx2
            e = [ e , [index ; index+nx1+1 ; index+1] , [index ; index+nx1 ; index+nx1+1 ] ];
        end
     
        index = index + 1;
    end
end
