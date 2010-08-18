static char help[] = "My first own testing utility for PETSc\n\n";

#include <iostream>
#include "petscksp.h"
#include "petscmat.h"
#include "petscmg.h"
#include "fem.h"
#include "solver.h"
#include "feti.h"

PetscScalar funConst(Point n) {
	return 1;
}

PetscScalar funSin(Point n) {
	return sin(n.x + n.y);
}

PetscScalar funStep(Point n) {

	PetscScalar c =  n.x>2?-1:1;
	c *=  n.y>2?-1:1;
	return c;
}

bool cf(PetscInt itNumber, PetscScalar rNorm, Vec *r) {
	PetscPrintf(PETSC_COMM_SELF, "%d - %e\n", itNumber, rNorm);
	return itNumber >	5;
}

int main(int argc, char *argv[]) {
	PetscScalar (*fList[])(Point) = {funConst, funSin, funStep};
	PetscErrorCode 	ierr;
	PetscMPIInt			rank, size;
	PetscReal				m=0.0,n=1.0,k=0.0,l=1.0,h=0.1;
  PetscInitialize(&argc,&argv,0,help);
	PetscInt				f = 2;
	PetscTruth 			flg;
	char fileName[PETSC_MAX_PATH_LEN]="matlab/out.m";			
	
	PetscOptionsGetReal("-test_m", &m, PETSC_NULL);
	PetscOptionsGetReal("-test_n", &n, PETSC_NULL);
	PetscOptionsGetReal("-test_k", &k, PETSC_NULL);
	PetscOptionsGetReal("-test_l", &l, PETSC_NULL);
	PetscOptionsGetReal("-test_h", &h, PETSC_NULL);
	PetscOptionsGetInt("-test_f", &f, PETSC_NULL);
	PetscOptionsGetString(PETSC_NULL, "-test_out_file", fileName, PETSC_MAX_PATH_LEN-1, &flg);
	//if (!flg) SETERRQ(1,"Must indicate binary file with the -test_out_file option");

	{
		ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
		MPI_Comm_size(PETSC_COMM_WORLD, &size);
		PetscViewer v;
		Mesh *mesh = new Mesh();
		mesh->generateRectangularMesh(m, n, k, l, h);
		mesh->partition(size);
		
		DistributedMesh *dm = new DistributedMesh();
		mesh->tear(dm);
		delete mesh;
		PetscViewerBinaryOpen(PETSC_COMM_WORLD, "matlab/mesh.m", FILE_MODE_WRITE, &v);
		dm->dumpForMatlab(v);
		PetscViewerDestroy(v);

		Feti1 feti(dm,fList[f], fList[0]);
		delete dm;

		PetscViewerBinaryOpen(PETSC_COMM_WORLD, fileName, FILE_MODE_WRITE, &v);
		feti.dumpSystem(v);
		feti.solve();
		feti.dumpSolution(v);
		PetscViewerDestroy(v);

	  
	//	delete mesh;
	}
	
	ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
