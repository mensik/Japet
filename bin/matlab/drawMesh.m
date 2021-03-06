function drawMesh(x,e,dirch,dual,primal)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

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

for i = 1:numEl
    a = x(e(i,1)+1,:);
    b = x(e(i,2)+1,:);
    c = x(e(i,3)+1,:);
    
    colors=['b', 'r', 'm','g','y'];
    smIndex = e(i,4) + 1;
    
    color = colors(mod(smIndex,5) + 1);
    %color = [colorA(smIndex) 0 colorB(smIndex)];
    line([a(1) b(1)], [a(2) b(2)], [a(3) b(3)], 'Color', color);
    line([a(1) c(1)], [a(2) c(2)], [a(3) c(3)], 'Color', color);
    line([b(1) c(1)], [b(2) c(2)], [b(3) c(3)], 'Color', color);
end
for i = 1:numPrimal
   line(x(primal(i)+1,1), x(primal(i)+1,2), x(primal(i)+1,3),'Marker','s',...
                                                       'MarkerFaceColor','r',...
                                                       'MarkerEdgeColor','r',...
                                                       'MarkerSize',9);
end
for i = 1:numDual
   line(x(dual(i)+1,1), x(dual(i)+1,2), x(dual(i)+1,3),'Marker','o',...
                                                       'MarkerFaceColor','green',...
                                                       'MarkerEdgeColor','green',...
                                                       'MarkerSize',4);
end
for i = 1:numDirch
   line(x(dirch(i)+1,1), x(dirch(i)+1,2), x(dirch(i)+1,3),'Marker','o','MarkerFaceColor','black');
end

end

