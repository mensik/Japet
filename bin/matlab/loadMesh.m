function [x,e,dirch,dual,primal] = loadMesh(filename)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

[x,e,dirch, dual, primal] = PetscBinaryRead(filename);
x = full(x);
e = full(e);
end

