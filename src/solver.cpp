#include "solver.h"

bool isConFun(PetscInt itNumber, PetscReal rNorm, Vec *r) {
	return rNorm < 1e-6;
}

void CGSolver::applyMult(Vec in, Vec out) {
	MatMult(A, in, out);
}

void CGSolver::initSolver(Vec b, Vec x) {
	this->b = b;
	this->x = x;

	sCtr = this;

	VecDuplicate(b, &r);
	VecDuplicate(b, &temp);
	VecDuplicate(r, &p);

	VecCopy(b, r);

	sApp->applyMult(x, temp);
	//MatMult(A, x, temp);
	VecAYPX(r, -1, temp);
	VecCopy(r, p);

	VecNorm(r, NORM_2, &rNorm);

	itCounter = 0;
	isCon = isConFun;

}

bool CGSolver::isConverged(PetscInt itNumber, PetscReal rNorm, Vec *r) {
	return isCon(itNumber, rNorm, r);
}

CGSolver::CGSolver(SolverApp *sa, Vec b, Vec x) {
	sApp = sa;
	initSolver(b, x);
}

CGSolver::CGSolver(Mat A, Vec b, Vec x) {
	this->A = A;
	sApp = this;
	initSolver(b, x);
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
		VecDot(p, temp, &pAp);
		PetscReal a = (rNorm * rNorm) / pAp;
		VecAXPY(x, -a, p);
		VecAXPY(r, -a, temp);

		PetscReal rNormS;
		VecNorm(r, NORM_2, &rNormS);

		PetscReal b = (rNormS * rNormS) / (rNorm * rNorm);
		VecAYPX(p, b, r);

		//PetscPrintf(PETSC_COMM_WORLD,"It: %d \t Res: %e\n",itCounter, rNorm);

		rNorm = rNormS;

	}

}

MPRGP::MPRGP(Mat A, Vec b, Vec l, Vec x, PetscReal G, PetscReal alp) {
	this->A = A;
	sApp = this;
	initSolver(b, l, x, G, alp);
}

MPRGP::MPRGP(SolverApp *app, Vec b, Vec l, Vec x, PetscReal G, PetscReal alp) {
	sApp = app;
	initSolver(b, l, x, G, alp);
}

MPRGP::~MPRGP() {
	VecDestroy(g);
	VecDestroy(p);
	VecDestroy(temp);
}

void MPRGP::initSolver(Vec b, Vec l, Vec x, PetscReal G, PetscReal alp) {
	this->b = b;
	this->x = x;
	this->l = l;
	this->G = G;
	this->alp = alp;

	e = 1e-8;

	VecGetOwnershipRange(x, &localRangeStart, &localRangeEnd);
	localRangeSize = localRangeEnd - localRangeStart;

	sCtr = this;
	sPC = NULL;

	VecDuplicate(b, &g);
	VecDuplicate(g, &p);
	VecDuplicate(g, &temp);
}

void MPRGP::applyMult(Vec in, Vec out) {
	MatMult(A, in, out);
}

void MPRGP::pcAction(Vec free, Vec z) {
	if (sPC != NULL) {
		sPC->applyPC(free, z);
		PetscReal *xArr, *lArr, *zArr;

		VecGetArray(z, &zArr);
		VecGetArray(x, &xArr);
		VecGetArray(l, &lArr);

		for (int i = 0; i < localRangeSize; i++)
			if (fabs(xArr[i] - lArr[i]) < 1e-8) { //active set
				zArr[i] = 0;
			}

		VecRestoreArray(z, &zArr);
		VecRestoreArray(x, &xArr);
		VecRestoreArray(l, &lArr);
	} else {
		VecCopy(free, z);
	}
}

bool MPRGP::isConverged(PetscInt itNum, PetscReal rNorm, Vec *vec) {
	return rNorm < 1e-3;
}

void MPRGP::solve() {

	Vec freeG, chopG, rFreeG, Ap, gp, z;
	VecDuplicate(g, &chopG);
	VecDuplicate(g, &freeG);
	VecDuplicate(g, &rFreeG);
	VecDuplicate(g, &Ap);
	VecDuplicate(g, &gp);
	VecDuplicate(g, &z);
	PetscReal normGP;
	PetscReal normCHG;

	projectFeas(x);

	VecCopy(b, g); //g = b
	sApp->applyMult(x, temp);
	VecAYPX(g, -1, temp); // g = A*x - g

	partGradient(freeG, chopG, rFreeG);

	VecCopy(freeG, gp);
	VecAYPX(gp, 1, chopG);
	VecNorm(gp, NORM_2, &normGP);
	VecNorm(chopG, NORM_2, &normCHG);

	itCounter = 0;

	pcAction(freeG, z);
	VecCopy(z, p);

	while (!sCtr->isConverged(itCounter, normGP, &x)) {
		PetscReal freeXrFree;
		VecDot(freeG, rFreeG, &freeXrFree);

		if (normCHG * normCHG <= G * G * freeXrFree) {
			//Trial conjugate gradient step
			sApp->applyMult(p, Ap);

			PetscReal gXp, pAp;
			VecDot(z, g, &gXp);
			VecDot(p, Ap, &pAp);

			PetscReal alpCG = gXp / pAp;
			PetscReal alpF = alpFeas();

			if (alpCG < alpF) {
				//CG step
				VecAXPY(x, -alpCG, p);
				VecAXPY(g, -alpCG, Ap);

				partGradient(freeG, chopG, rFreeG);
				pcAction(freeG, z);
				PetscReal freeGAp;
				VecDot(z, Ap, &freeGAp);
				PetscReal beta = freeGAp / pAp;

				VecAYPX(p, -beta, z);
			} else {
				//Expansion step
				VecAXPY(x, -alpF, p);
				VecAXPY(g, -alpF, Ap);
				partGradient(freeG, chopG, rFreeG);

				VecAXPY(x, -alp, freeG);
				projectFeas(x);
				VecCopy(b, g);
				sApp->applyMult(x, temp);
				VecAYPX(g, -1, temp);

				partGradient(freeG, chopG, rFreeG);
				pcAction(freeG, z);

				VecCopy(z, p);
			}
		} else {
			//Proportioning step
			PetscReal dg, dAd;
			sApp->applyMult(chopG, Ap);
			VecDot(g, chopG, &dg);
			VecDot(chopG, Ap, &dAd);

			PetscReal alpCG = dg / dAd;

			VecAXPY(x, -alpCG, chopG);
			VecAXPY(g, -alpCG, Ap);

			partGradient(freeG, chopG, rFreeG);
			pcAction(freeG, z);
			VecCopy(z, p);
		}
		VecCopy(freeG, gp);
		VecAYPX(gp, 1, chopG);
		VecNorm(gp, NORM_2, &normGP);
		VecNorm(chopG, NORM_2, &normCHG);
		itCounter++;
	}

	VecDestroy(chopG);
	VecDestroy(freeG);
	VecDestroy(rFreeG);
	VecDestroy(gp);
	VecDestroy(Ap);
}

void MPRGP::partGradient(Vec &freeG, Vec &chopG, Vec &rFreeG) {
	VecZeroEntries(freeG);
	VecZeroEntries(chopG);
	VecZeroEntries(rFreeG);

	PetscReal *xArr, *lArr, *gArr, *freeGArr, *chopGArr, *rFreeGArr;

	VecGetArray(g, &gArr);
	VecGetArray(freeG, &freeGArr);
	VecGetArray(chopG, &chopGArr);
	VecGetArray(rFreeG, &rFreeGArr);
	VecGetArray(x, &xArr);
	VecGetArray(l, &lArr);

	for (int i = 0; i < localRangeSize; i++) {
		if (fabs(xArr[i] - lArr[i]) < 1e-8) { //active set
			if (gArr[i] < 0) {
				chopGArr[i] = gArr[i];
			}
		} else { //free set
			freeGArr[i] = gArr[i];
			PetscReal rG = (xArr[i] - lArr[i]) / alp;
			if (rG < gArr[i]) {
				rFreeGArr[i] = rG;
			} else {
				rFreeGArr[i] = gArr[i];
			}
		}
	}

	VecRestoreArray(g, &gArr);
	VecRestoreArray(freeG, &freeGArr);
	VecRestoreArray(chopG, &chopGArr);
	VecRestoreArray(rFreeG, &rFreeGArr);
	VecRestoreArray(x, &xArr);
	VecRestoreArray(l, &lArr);
}

void MPRGP::projectFeas(Vec &vec) {
	PetscReal *lArr, *vecArr;
	VecGetArray(l, &lArr);
	VecGetArray(vec, &vecArr);

	for (int i = 0; i < localRangeSize; i++) {
		if (lArr[i] > vecArr[i]) vecArr[i] = lArr[i];
	}

	VecRestoreArray(l, &lArr);
	VecRestoreArray(vec, &vecArr);
}

PetscReal MPRGP::alpFeas() {
	PetscFunctionBegin;
	PetscReal alpF;
	PetscReal *lArr, *xArr, *pArr;
	VecGetArray(l, &lArr);
	VecGetArray(x, &xArr);
	VecGetArray(p, &pArr);
	alpF = 1e15;

	for (int i = 0; i < localRangeSize; i++)
		if (pArr[i] > 0) {
			PetscReal a = (xArr[i] - lArr[i]) / pArr[i];
			if (a < alpF) alpF = a;
		}

	VecRestoreArray(l, &lArr);
	VecRestoreArray(x, &xArr);
	VecRestoreArray(p, &pArr);

	PetscReal alpFGlobal;
	MPI_Allreduce(&alpF, &alpFGlobal, 1, MPI_DOUBLE, MPI_MIN, PETSC_COMM_WORLD);

	PetscFunctionReturn(alpFGlobal);
}

int MPRGP::getNumIterations() {
	return itCounter;
}

