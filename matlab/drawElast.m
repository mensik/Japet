function drawElast(x,e,u,dirch,dual,primal)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

magnify = 6e3;

numEl = size(e,1);
numDirch = size(dirch,1);
numDual = size(dual,1);
numPrimal = size(primal,1);

procCount = max(x(:,4));
colorA = 0:1/procCount:1;
colorB = 1:-1/procCount:0;

%for i = 1:numVe
%    line(x(i,1), x(i,2), x(i,3),'Marker','o');
%end
hold on
for i = 1:numEl
    a = x(e(i,1)+1,:);
    b = x(e(i,2)+1,:);
    c = x(e(i,3)+1,:);
    
    colors=[[1 1 1]; [0.9 0.9 0.9]];
    smIndex = e(i,4) + 1;
    
    color = colors(mod(smIndex,2) + 1, :);
    %color = [colorA(smIndex) 0 colorB(smIndex)];
    
    dX = u(2*e(i,[1 2 3 1])+[1 1 1 1]);
    dY = u(2*e(i,[1 2 3 1])+[2 2 2 2]);
    X = [a(1) b(1) c(1) a(1)];
    Y = [a(2) b(2) c(2) a(2)];

    fill(X + dX'*magnify,Y + dY'*magnify, color);
    %line(X + dX'*magnify,Y + dY'*magnify, 'Color', 'k');
end
     
for i = 1:numDirch
   line(x(dirch(i)+1,1), x(dirch(i)+1,2), x(dirch(i)+1,3),'Marker','o','MarkerFaceColor','k', 'MarkerEdgecolor','k');
end

daspect([1 1 1])
