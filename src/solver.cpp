#include "solver.h"

void SolverApp::setRequiredPrecision(PetscReal reqPrecision) {

}

Solver::Solver(Mat A, Vec b, Vec x) {

	this->A = A;
	this->b = b;
	this->x = x;

	sApp = this;

	init();
}

Solver::Solver(SolverApp *sa, Vec b, Vec x, SolverPreconditioner *PC) {
	this->sApp = sa;
	this->b = b;
	this->x = x;

	if (PC == NULL) {
		this->sPC = this;
	} else {
		this->sPC = PC;
	}

	init();
}

Solver::~Solver() {

	VecDestroy(g);
}

void Solver::init() {

	sCtr = this;
	precision = 1e-3;
	VecNorm(b, NORM_2, &bNorm);

	VecDuplicate(b, &g);
	VecDuplicate(b, &z);

	Vec temp;
	VecDuplicate(b, &temp);

	VecCopy(b, g);

	sApp->setRequiredPrecision(MAXPREC);
	sApp->applyMult(x, temp, &itManager);
	VecAYPX(g, -1, temp);

	VecDestroy(temp);
	sPC->applyPC(g, z);

	VecDot(g, z, &rNorm);
	rNorm = sqrt(rNorm);

	r0Norm = rNorm;
}

void Solver::applyMult(Vec in, Vec out, IterationManager *info) {
	MatMult(A, in, out);
}

void Solver::applyPC(Vec r, Vec z) {
	VecCopy(r, z);
}

bool Solver::isConverged(PetscInt itNumber, PetscReal rNorm, PetscReal bNorm,
		Vec *x) {
	return rNorm / bNorm < precision;
}

void Solver::nextIteration() {
	itManager.setIterationData("!normG", rNorm);
	itManager.setIterationData("R", pow(rNorm / r0Norm, 2.0 / (double)itManager.getItCount()) );
	itManager.nextIteration();
}

void RichardsSolver::solve() {

	Vec Ax;
	VecDuplicate(x, &Ax);

	while (!sCtr->isConverged(getItCount(), rNorm, bNorm, &g)) {
		nextIteration();

		VecAXPY(x, -alpha, g);

		sApp->setRequiredPrecision(rNorm * 1e-3);

		sApp->applyMult(x, Ax);

		VecCopy(b, g);
		VecAYPX(g, -1, Ax);
		VecNorm(g, NORM_2, &rNorm);
	}

	VecDestroy(Ax);
}

void CGSolver::initSolver() {
	VecDuplicate(b, &temp);
	VecDuplicate(g, &p);
	VecCopy(z, p);
	setTitle("CG");
}

CGSolver::~CGSolver() {
	VecDestroy(p);
	VecDestroy(temp);
}

void CGSolver::solve() {

	PetscReal xNorm = 0.001;
	//VecNorm(x, NORM_2, &xNorm);
	while (!sCtr->isConverged(getItCount(), rNorm, xNorm, &g)) {

		nextIteration();

		PetscScalar pAp;
		sApp->applyMult(p, temp, &itManager);
		VecDot(p, temp, &pAp);

		PetscReal a = (rNorm * rNorm) / pAp;
		VecAXPY(x, -a, p);
		VecAXPY(g, -a, temp);

		sPC->applyPC(g, z);

		PetscReal gDOTz;
		VecDot(g, z, &gDOTz);


		PetscReal beta = gDOTz / (rNorm * rNorm);
		VecAYPX(p, beta, z);

		rNorm = sqrt(gDOTz);
		VecNorm(x, NORM_2,  &xNorm);
	}
}

void ASinStep::initSolver() {
	tau = 1;
	fi = (sqrt((double) 5.0) - 1) / 2;
	z = 0;
	setTitle("Arcsine Density Descent");
}

void ASinStep::solve() {
	PetscReal beta;
	Vec Ag, Ax;
	VecDuplicate(g, &Ag);
	VecDuplicate(x, &Ax);

	int gradRestartLoop = 2;
	PetscReal outNorm = 1e-1;


	for (int i = 0; i < 2; i++) {

		sApp->setRequiredPrecision(outNorm);

		sApp->applyMult(g, Ag, &itManager);

		PetscReal gg, gAg;
		VecDot(g, g, &gg);
		VecDot(Ag, g, &gAg);
		beta = gAg / gg;

		if (getItCount() == 0) {
			m = beta;
			M = beta;
		} else {
			m = fmin(m, beta);
			M = fmax(M, beta);
		}

		VecAXPY(x, -1 / beta, g);
		VecAXPY(g, -1 / beta, Ag);

		setIterationData("m", m);
		setIterationData("M", M);
		setIterationData("beta", beta);
		setIterationData("reqPrec", outNorm);

		nextIteration();
		VecNorm(g, NORM_2, &rNorm);
	}
	while (!sCtr->isConverged(getItCount(), rNorm, bNorm, &g)) {

		outNorm = outNorm * 0.66;

		PetscReal reqPrec = fmax(fmin(outNorm, rNorm * 1e-1), 1e-6);
		sApp->setRequiredPrecision(reqPrec);
		if (reqPrec < pow((float)10,(int)-1*(gradRestartLoop))) {
			gradRestartLoop++;
		}
		setIterationData("reqPrec", reqPrec);


		if (getItCount() % gradRestartLoop == 0) {
			//sApp->applyMult(x, Ax);
		} else {
			sApp->applyMult(g, Ag, &itManager);

			PetscReal gg, gAg, moment1;
			VecDot(g, g, &gg);
			VecDot(Ag, g, &gAg);
			moment1 = gAg / gg;
			m = fmin(m, moment1);
			M = fmax(M, moment1);
		}

		if (getItCount() % 2 == 0) {
			PetscReal eps = tau * (M - m);
			z += fi;
			beta = m + eps + (std::cos(PI * z) + 1) * (M - m - 2 * eps) / 2;
		} else {
			beta = M + m - beta;
		}

		VecAXPY(x, -1 / beta, g);

		if (getItCount() % gradRestartLoop == 0) {
			VecCopy(b, g);
			sApp->applyMult(x, Ax, &itManager);
			VecAYPX(g, -1, Ax);
			VecNorm(g, NORM_2, &rNorm);
		} else {
			VecAXPY(g, -1 / beta, Ag);
			VecNorm(g, NORM_2, &rNorm);
		}

		setIterationData("m", m);
		setIterationData("M", M);
		setIterationData("beta", beta);

		nextIteration();

	}

	VecDestroy(Ag);
	VecDestroy(Ax);
}

MPRGP::~MPRGP() {
	;
	VecDestroy(p);
	VecDestroy(temp);
}

void MPRGP::initSolver(Vec l, PetscReal G, PetscReal alp) {
	this->l = l;
	this->G = G;
	this->alp = alp;

	e = 1e-8;

	VecGetOwnershipRange(x, &localRangeStart, &localRangeEnd);
	localRangeSize = localRangeEnd - localRangeStart;

	sPC = NULL;
	VecDuplicate(g, &p);
	VecDuplicate(g, &temp);

	setTitle("MPRGP");
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

	pcAction(freeG, z);
	VecCopy(z, p);

	while (!sCtr->isConverged(getItCount(), rNorm, bNorm, &x)) {
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
				setIterationData("stepType", CG);
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
				setIterationData("stepType", Expansion);
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
			setIterationData("stepType", Proportion);
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

		VecDot(gp, z, &rNorm);
		rNorm = sqrt(rNorm);

		setIterationData("normGP", normGP);
		nextIteration();
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
		if (lArr[i] > vecArr[i]) {
			vecArr[i] = lArr[i];
		}
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

