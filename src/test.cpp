static char help[] = "My first own testing utility for PETSc\n\n";

#include <iostream>
#include "structures.h"

int main(int argc, char *argv[]) {
	PetscReal				m=0.0,n=1.0,k=0.0,l=1.0,h=0.001;

	PetscInitialize(&argc,&argv,(char *)0,help);
		
	Mesh mesh;
	mesh.generateRectangularMesh(m, n, k, l, h); 
/*
	PetscPrintf(PETSC_COMM_WORLD, "%d, %d\n", mesh.elements.size(), mesh.vetrices.size());
	
	PetscPrintf(PETSC_COMM_WORLD, "*Elements*\n");
	for (std::map<PetscInt, Element>::iterator i = mesh.elements.begin(); i != mesh.elements.end(); i++) {
		PetscPrintf(PETSC_COMM_WORLD, "%d: %d-%d-%d\n", i->first, i->second.vetrices[0], i->second.vetrices[1], i->second.vetrices[2]);
	}
	
	PetscPrintf(PETSC_COMM_WORLD, "*Edges*\n");
	for (std::map<PetscInt, Edge>::iterator i = mesh.edges.begin(); i != mesh.edges.end(); i++) {
		PetscPrintf(PETSC_COMM_WORLD, "%d: %d-%d\n", i->first, i->second.vetrices[0], i->second.vetrices[1]);
	}

	PetscPrintf(PETSC_COMM_WORLD, "*Border*\n");
	for (std::set<PetscInt>::iterator i = mesh.borderEdges.begin(); i != mesh.borderEdges.end(); i++) {
		PetscPrintf(PETSC_COMM_WORLD, "%d ", *i);
	}
	PetscPrintf(PETSC_COMM_WORLD, "\n");
*/
	PetscFinalize();
  return 0;
}
