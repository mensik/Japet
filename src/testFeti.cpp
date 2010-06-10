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
	PetscMPIInt			rank;
	PetscReal				m=0.0,n=4.0,k=0.0,l=4.0,h=0.1;
	PetscInt				xSubs = 4, ySubs = 4;
  PetscInitialize(&argc,&argv,0,help);
	PetscInt				f = 2;
	PetscInt				boundedSideCount = 4;
	PetscTruth 			flg;
	char fileName[PETSC_MAX_PATH_LEN]="matlab/out.m";			
	
	PetscOptionsGetReal("-test_h", &h, PETSC_NULL);
	PetscOptionsGetInt("-test_x", &xSubs, PETSC_NULL);
	PetscOptionsGetInt("-test_y", &ySubs, PETSC_NULL);
	PetscOptionsGetInt("-test_f", &f, PETSC_NULL);
	PetscOptionsGetInt("-test_bounded_side_count", &boundedSideCount, PETSC_NULL);
	PetscOptionsGetString(PETSC_NULL, "-test_out_file", fileName, PETSC_MAX_PATH_LEN-1, &flg);
	//if (!flg) SETERRQ(1,"Must indicate binary file with the -test_out_file option");

	{
		ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
		BoundSide dirchlet[] = {LEFT,RIGHT,TOP, BOTTOM};
		PetscViewer v;
		PetscViewerBinaryOpen(PETSC_COMM_WORLD, fileName, FILE_MODE_WRITE, &v);
		Mesh *mesh;
		generateRectangularTearedMesh(m,n,k,l,h,xSubs,ySubs,boundedSideCount,dirchlet, &mesh);
		Feti1 feti(mesh,fList[f], fList[0]);
		feti.dumpSystem(v);
		feti.solve();
		feti.dumpSolution(v);
		PetscViewerDestroy(v);

		PetscViewerBinaryOpen(PETSC_COMM_WORLD, "matlab/mesh.m", FILE_MODE_WRITE, &v);
		mesh->dumpForMatlab(v);
		PetscViewerDestroy(v);
	  
		delete mesh;
	}
	
	ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
