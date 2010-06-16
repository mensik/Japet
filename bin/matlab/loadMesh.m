function [x,e,dirch,dual,primal] = loadMesh(filename)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

[x,e] = PetscBinaryRead(filename);
x = full(x);
e = full(e);
dirch = [];
dual = [];
primal = [];
end

