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

MPRGP::MPRGP(Mat A, Vec b, Vec l, Vec x, PetscReal G, PetscReal alp) {
	this->A = A;
	this->b = b;
	this->x = x;
	this->l = l;
	this->G = G;
	this->alp = alp;

	e = 1e-6;

	VecGetOwnershipRange(x, &localRangeStart, &localRangeEnd);
	localRangeSize = localRangeEnd - localRangeStart;

	PetscMalloc(localRangeSize*sizeof(PetscInt), &localIndices);
	for (int i = localRangeStart; i < localRangeEnd; i++) localIndices[i - localRangeStart] = i;

	VecDuplicate(b, &g);
	VecDuplicate(b, &temp);
	VecDuplicate(g,&p);
	
	projectFeas(x);

	VecCopy(b,g);					//g = b
	MatMult(A, x, temp);
	VecAYPX(g, -1, temp); // g = A*x - g
}

MPRGP::~MPRGP() {
	PetscFree(localIndices);
}

void MPRGP::solve() {

	Vec freeG, chopG;
	VecDuplicate(g, &chopG);
	VecDuplicate(g, &freeG);
	PetscReal normFG, normCHG;

	PetscInt itCounter = 0;	
	do {
		itCounter++;
		VecAXPY(x, -alp, g);
		projectFeas(x);

		VecCopy(b,g);					//g = b
		MatMult(A, x, temp);
		VecAYPX(g, -1, temp); // g = A*x - g		
		
		partGradient(freeG, chopG);
		VecNorm(freeG, NORM_2, &normFG);
		VecNorm(chopG, NORM_2, &normCHG);

		PetscPrintf(PETSC_COMM_WORLD, "%d: %f\n", itCounter, normFG + normCHG);
	} while (normFG + normCHG > 1e-3);
}

void MPRGP::partGradient(Vec &freeG, Vec &chopG) {
	VecZeroEntries(freeG);
	VecZeroEntries(chopG);

	Vec difV;
	VecDuplicate(x, &difV);
	VecWAXPY(difV, -1, x, l);
	
	PetscScalar *diff, *gArr, *freeGArr, *chopGArr;

	VecGetArray(g, &gArr);
	VecGetArray(freeG, &freeGArr);
	VecGetArray(chopG, &chopGArr);
	VecGetArray(difV, &diff);

	for (int i = 0; i < localRangeSize; i++) {
		if (abs(diff[i]) < e) { //active set
			if (gArr[i] < 0) {
				chopGArr[i] = gArr[i];
			}
		} else { 								//free set
			freeGArr[i] = gArr[i];
		}
	}

	VecRestoreArray(g, &gArr);
	VecRestoreArray(freeG, &freeGArr);
	VecRestoreArray(chopG, &chopGArr);
	VecRestoreArray(difV, &diff);

	VecDestroy(difV);
}

void MPRGP::projectFeas(Vec &vec) {
	PetscScalar *lArr, *vecArr;
	VecGetArray(l, &lArr);
	VecGetArray(vec, &vecArr);
	
	for (int i = 0; i < localRangeSize; i++) {
		if (lArr[i] > vecArr[i]) vecArr[i] = lArr[i];	
	}

	VecRestoreArray(l, &lArr);
	VecRestoreArray(vec, &vecArr);
}
