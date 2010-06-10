static char help[] = "My first own testing utility for PETSc\n\n";

#include <iostream>
#include "petscksp.h"
#include "petscmat.h"
#include "petscmg.h"
#include "structures.h"
#include "fem.h"
#include "solver.h"
#include "feti.h"


PetscScalar funConst(Point n) {
	return 1;
}

PetscScalar funSin(Point n) {
	return sin(n.x + n.y);
}

void applyMat(Vec x) {
	VecSet(x, 13);
}

int main(int argc, char *argv[]) {
	PetscScalar (*fList[])(Point) = {funConst, funSin};
	PetscInt			f=0,K=0;
	PetscErrorCode 	ierr;
	Mesh			*mesh;
	PetscReal				m=0.0,n=4.0,k=0.0,l=4.0,h=0.5;
	Mat							A;
	Vec							b;
  Vec 						x;

	PetscInitialize(&argc,&argv,(char *)0,help);
		
	BoundSide dirchlet[] = {LEFT,TOP,RIGHT,BOTTOM};
	
	generateRectangularTearedMesh(m,n,k,l,h,2,1,1,dirchlet, &mesh);
	FEMAssemble2DLaplace(PETSC_COMM_WORLD, mesh,A,b,fList[f],fList[K]);

	PetscInt n_rows, n_cols;
	MatGetSize(A, &n_rows, &n_cols);
	VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, n_rows, &x);

	KSP ksp;
	KSPCreate(PETSC_COMM_WORLD, &ksp);
	KSPSetOperators(ksp, A, A, SAME_PRECONDITIONER);

	MatNullSpace ns;
	MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, PETSC_NULL, &ns);
	//MatNullSpaceRemove(ns, b, PETSC_NULL);

	KSPSetNullSpace(ksp, ns);

	KSPSolve(ksp, b, x);

	PetscViewer v;
	PetscViewerBinaryOpen(PETSC_COMM_WORLD, "matlab/out.m" , FILE_MODE_WRITE, &v);

	MatView(A,v);
	VecView(b,v);
	VecView(x,v);

	PetscViewerDestroy(v);
	MatDestroy(A);
	VecDestroy(b);
	VecDestroy(x);
	KSPDestroy(ksp);

	ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
