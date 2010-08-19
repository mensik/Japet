static char help[] = "My first own testing utility for PETSc\n\n";

#include <iostream>
#include "petscksp.h"
#include "petscmat.h"
#include "petscmg.h"
#include "fem.h"
#include "solver.h"
#include "smale.h"
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
	return itNumber > 	5; 
}

int main(int argc, char *argv[]) {
	PetscScalar (*fList[])(Point) = {funConst, funSin, funStep};
	PetscErrorCode 	ierr;
	PetscMPIInt			rank,size;
	PetscReal				m=0.0,n=4.0,k=0.0,l=4.0,h=0.1;
	PetscInt				xSubs = 4, ySubs = 4;
	PetscReal				mi=1e-3,ro=4,beta=1.5,M=1;
  PetscInitialize(&argc,&argv,0,help);
	PetscInt				f = 2;
	PetscTruth 			flg;

	PetscLogStage		assemblyStage, smaleStage;


	char fileName[PETSC_MAX_PATH_LEN]="matlab/out.m";			
	
	PetscOptionsGetReal("-test_h", &h, PETSC_NULL);
	PetscOptionsGetReal("-test_mi", &mi, PETSC_NULL);
	PetscOptionsGetReal("-test_ro", &ro, PETSC_NULL);
	PetscOptionsGetReal("-test_beta", &beta, PETSC_NULL);
	PetscOptionsGetReal("-test_M", &M, PETSC_NULL);
	PetscOptionsGetInt("-test_x", &xSubs, PETSC_NULL);
	PetscOptionsGetInt("-test_y", &ySubs, PETSC_NULL);
	PetscOptionsGetInt("-test_f", &f, PETSC_NULL);
	PetscOptionsGetString(PETSC_NULL, "-test_out_file", fileName, PETSC_MAX_PATH_LEN-1, &flg);
	//if (!flg) SETERRQ(1,"Must indicate binary file with the -test_out_file option");

	{
  	//n = xSubs;
		//l = ySubs;
		
		ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
		MPI_Comm_size(PETSC_COMM_WORLD, &size);
		PetscViewer v;

		PetscLogStageRegister("Assembly", &assemblyStage);
		PetscLogStagePush(assemblyStage);
		Mesh *mesh = new Mesh();
		mesh->generateRectangularMesh(m, n, k, l, h);
		mesh->partition(size);
		DistributedMesh *dm = new DistributedMesh();
		mesh->tear(dm);
		delete mesh;
		SDSystem sdSystem(dm,fList[f],fList[0]);
		PetscViewerBinaryOpen(PETSC_COMM_WORLD, "matlab/mesh.m", FILE_MODE_WRITE, &v);
		dm->dumpForMatlab(v);
		PetscViewerDestroy(v);
		delete dm;
		PetscLogStagePop();
		
		Smale smale(&sdSystem,mi,ro,beta,M);
		PetscLogStageRegister("Smale", &smaleStage);
		PetscLogStagePush(smaleStage);
		smale.solve();
		PetscLogStagePop();
		PetscViewerBinaryOpen(PETSC_COMM_WORLD, fileName, FILE_MODE_WRITE, &v);
		smale.dumpSolution(v);
		//smale.dumpSolution(matLabView);
		PetscViewerDestroy(v);

		//PetscViewerDestroy(matLabView);
	}
	
	ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
