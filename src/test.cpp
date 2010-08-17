static char help[] = "My first own testing utility for PETSc\n\n";

#include <iostream>
#include "structures.h"
#include "petscmat.h"

int main(int argc, char *argv[]) {
	PetscReal				m=0.0,n=1.0,k=0.0,l=1.0,h=0.5;

	PetscInitialize(&argc,&argv,(char *)0,help);
	PetscInt size,rank;
	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
	MPI_Comm_size(PETSC_COMM_WORLD, &size);
	Mesh mesh;
	mesh.generateRectangularMesh(m, n, k, l, h);
	mesh.partition(size);


	DistributedMesh dm;
	mesh.tear(&dm);
	
	PetscSynchronizedPrintf(PETSC_COMM_WORLD, "[%d] Nodes on this proces: %d \n", rank, dm.nVetrices);
	for (std::set<PetscInt>::iterator i = dm.indDirchlet.begin(); i != dm.indDirchlet.end(); i++) {
		PetscSynchronizedPrintf(PETSC_COMM_WORLD, "\t%d\n", *i);
	}
	PetscSynchronizedFlush(PETSC_COMM_WORLD);
	
	//ISView(dm.oldOrdering, PETSC_VIEWER_STDOUT_WORLD);
	//ISView(dm.newOrdering, PETSC_VIEWER_STDOUT_WORLD);
	AOView(dm.procesOrdering, PETSC_VIEWER_STDOUT_WORLD);
	
	//mesh.save("domain.jpm", true);
	
	//Mesh mesh2;
	//mesh2.load("domain.jpm", false);

	PetscViewer v;
	
	PetscViewerBinaryOpen(PETSC_COMM_WORLD, "matlab/mesh.m", FILE_MODE_WRITE, &v);
	mesh.dumpForMatlab(v);
	PetscViewerDestroy(v);

	PetscFinalize();
  return 0;
}
