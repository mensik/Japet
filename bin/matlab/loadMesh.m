function [x,e,dirch,dual] = loadMesh(filename)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

[x,e,dirch,dual] = PetscBinaryRead(filename);
x = full(x);
e = full(e);

end

