#include "solver.h"

bool isConFun(PetscInt itNumber, PetscScalar rNorm, Vec *r) {
	return rNorm < 1e-6; 
}

void CGSolver::applyMult(Vec in,Vec out) {
	MatMult(A,in, out);
}

void CGSolver::initSolver(Vec b, Vec x) {
	this->b = b;
	this->x = x;

	sCtr = this;
		
	VecDuplicate(b, &r);
	VecDuplicate(b, &temp);
	VecDuplicate(r,&p);
	
	VecCopy(b,r);

	sApp->applyMult(x, temp);
	//MatMult(A, x, temp);
	VecAYPX(r,-1,temp);
	VecCopy(r,p);
	
	VecNorm(r, NORM_2, &rNorm);

	itCounter = 0;
	isCon = isConFun;

}

bool CGSolver::isConverged(PetscInt itNumber, PetscScalar rNorm, Vec *r) {
	return isCon(itNumber, rNorm, r);
}

CGSolver::CGSolver(SolverApp *sa, Vec b, Vec x) {
	sApp = sa;
	initSolver(b,x);
}

CGSolver::CGSolver(Mat A, Vec b, Vec x) {
	this->A = A;
	sApp = this;
	initSolver(b,x);
}

CGSolver::~CGSolver() {
	VecDestroy(p);
	VecDestroy(r);
	VecDestroy(temp);
}

void CGSolver::solve() {
	while (!sCtr->isConverged(itCounter, rNorm, &r)) {
	//	PetscPrintf(PETSC_COMM_WORLD, "%d - %e\n",itCounter, rNorm);
		itCounter++;
		PetscScalar pAp;
		//MatMult(A,p,temp);
		sApp->applyMult(p, temp);
		VecDot(p, temp,&pAp);
		PetscScalar a = (rNorm*rNorm) / pAp;
		VecAXPY(x,-a,p);
		VecAXPY(r,-a,temp);
		
		PetscReal rNormS;
		VecNorm(r, NORM_2, &rNormS);
		
		PetscScalar b = (rNormS * rNormS) / (rNorm * rNorm);
		VecAYPX(p,b,r);
	
		//PetscPrintf(PETSC_COMM_WORLD,"It: %d \t Res: %e\n",itCounter, rNorm);
			
		rNorm = rNormS;

	}

}
