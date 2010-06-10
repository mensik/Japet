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
	PetscMPIInt			rank;
	PetscReal				m=0.0,n=4.0,k=0.0,l=4.0,h=0.1;
	PetscInt				xSubs = 4, ySubs = 4;
	PetscReal				mi=1e-3,ro=4,beta=1.5,M=1;
  PetscInitialize(&argc,&argv,0,help);
	PetscInt				f = 2;
	PetscInt				boundedSideCount = 1;
	PetscTruth 			flg;

	PetscLogStage		assemblyStage, dirchletStage, smaleStage;


	char fileName[PETSC_MAX_PATH_LEN]="matlab/out.m";			
	
	PetscOptionsGetReal("-test_h", &h, PETSC_NULL);
	PetscOptionsGetReal("-test_mi", &mi, PETSC_NULL);
	PetscOptionsGetReal("-test_ro", &ro, PETSC_NULL);
	PetscOptionsGetReal("-test_beta", &beta, PETSC_NULL);
	PetscOptionsGetReal("-test_M", &M, PETSC_NULL);
	PetscOptionsGetInt("-test_x", &xSubs, PETSC_NULL);
	PetscOptionsGetInt("-test_y", &ySubs, PETSC_NULL);
	PetscOptionsGetInt("-test_f", &f, PETSC_NULL);
	PetscOptionsGetInt("-test_bounded_side_cont", &boundedSideCount, PETSC_NULL);
	PetscOptionsGetString(PETSC_NULL, "-test_out_file", fileName, PETSC_MAX_PATH_LEN-1, &flg);
	//if (!flg) SETERRQ(1,"Must indicate binary file with the -test_out_file option");

	{
  	n = xSubs;
		l = ySubs;
		
		ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
		
		PetscViewer v;
		PetscViewerBinaryOpen(PETSC_COMM_WORLD, fileName, FILE_MODE_WRITE, &v);

		//PetscViewer matLabView;
		//PetscViewerSocketOpen(PETSC_COMM_WORLD, PETSC_NULL, PETSC_DEFAULT, &matLabView);

		//PetscViewerASCIIOpen(PETSC_COMM_WORLD,fileName,&v);
		//PetscViewerSetFormat(v,PETSC_VIEWER_ASCII_MATLAB);

		PetscLogStageRegister("Assembly", &assemblyStage);
		PetscLogStagePush(assemblyStage);
		SDRectSystem sdSystem(m,n,k,l,h,xSubs,ySubs,fList[f],fList[0]);
		PetscLogStagePop();
		
		
		BoundSide dirchlet[] = {LEFT,TOP,RIGHT,BOTTOM};
		
		
		PetscLogStageRegister("Dirchlet", &dirchletStage);
		PetscLogStagePush(dirchletStage);
		sdSystem.setDirchletBound(boundedSideCount,dirchlet);
		PetscLogStagePop();
		
		Smale smale(&sdSystem,mi,ro,beta,M);
		PetscLogStageRegister("Smale", &smaleStage);
		PetscLogStagePush(smaleStage);
		smale.solve();
		PetscLogStagePop();
		smale.dumpSolution(v);
		//smale.dumpSolution(matLabView);
		PetscViewerDestroy(v);
		//PetscViewerDestroy(matLabView);
	}
	
	ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
