function [x,e] = prepareMesh(sizeX,sizeY,h)

subDomainsCount=sizeX*sizeY;

x = cell(subDomainsCount,1);
e = cell(subDomainsCount,1);

p = 1;

for j = 0:sizeY-1
    for i = 0:sizeX-1
        k = j*sizeX + i + 1;
        [x{k},e{k}] = discretRectangle(i,i+1,j,j+1,h,p);
        p = p+ size(x{k},2);
    end
end

x = cell2mat(x');
e = cell2mat(e');