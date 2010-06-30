static char help[] = "My first own testing utility for PETSc\n\n";

#include <iostream>
#include "structures.h"
#include "petscmat.h"

int main(int argc, char *argv[]) {
	PetscReal				m=0.0,n=1.0,k=0.0,l=1.0,h=0.5;

	PetscInitialize(&argc,&argv,(char *)0,help);
		
	Mesh mesh;
	mesh.generateRectangularMesh(m, n, k, l, h);
	mesh.partition(3);
	mesh.tear();
	mesh.save("domain.jpm", true);
	
	//Mesh mesh2;
	//mesh2.load("domain.jpm", false);

	//PetscViewer v;
	
	//PetscViewerBinaryOpen(PETSC_COMM_WORLD, "matlab/mesh.m", FILE_MODE_WRITE, &v);
	//mesh.dumpForMatlab(v);
	//PetscViewerDestroy(v);

	PetscFinalize();
  return 0;
}
