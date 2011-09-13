#include "solver.h"

void SolverApp::setRequiredPrecision(PetscReal reqPrecision) {

}

Solver::Solver(Mat A, Vec b, Vec x, SolverPreconditioner *PC, MPI_Comm comm) {

	this->A = A;
	this->b = b;
	this->x = x;
	this->comm = comm;

	sApp = this;

	if (PC == NULL) {
		this->sPC = this;
	} else {
		this->sPC = PC;
	}

	init();
}

Solver::Solver(SolverApp *sa, Vec b, Vec x, SolverPreconditioner *PC,
		MPI_Comm comm) {
	this->sApp = sa;
	this->b = b;
	this->x = x;
	this->comm = comm;

	if (PC == NULL) {
		this->sPC = this;
	} else {
		this->sPC = PC;
	}

	init();
}

Solver::~Solver() {

	VecDestroy(z);
	VecDestroy(g);
}

void Solver::init() {

	itManager.setComm(comm);

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
	VecAYPX(g, -1, temp); // g = Ax - b

	VecDestroy(temp);

	sPC->applyPC(g, z);

	VecNorm(g, NORM_2, &rNorm);
	r0Norm = rNorm;

}

void Solver::applyMult(Vec in, Vec out, IterationManager *info) {
	MatMult(A, in, out);
}

void Solver::applyPC(Vec r, Vec rz) {

	VecCopy(r, rz);
}

bool Solver::isConverged(PetscInt itNumber, PetscReal rNorm, PetscReal bNorm,
		Vec *x) {
	return rNorm / bNorm < precision;
}

void Solver::nextIteration() {
	itManager.setIterationData("!normG", rNorm);
	itManager.setIterationData("R", pow(rNorm / r0Norm, 2.0
			/ (double) itManager.getItCount()));
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

	if (bNorm > 1e-16) {
		PetscReal relativeCoef = MAX(bNorm, r0Norm);
		PetscReal gTzPrev, gTz;

		//solProj->applyProjection(g);
		//	sPC->applyPC(g, z);
		VecDot(g, z, &gTz);

		while (!sCtr->isConverged(getItCount(), rNorm, relativeCoef, &g)) {

			MyLogger::Instance()->getTimer("Iteration")->startTimer();
			itManager.setIterationData("Rel. err", rNorm / relativeCoef);
			nextIteration();

			PetscScalar pAp;
			sApp->applyMult(p, temp, &itManager);

			PetscReal tNorm;
			VecNorm(temp, NORM_2, &tNorm);

			VecDot(p, temp, &pAp);

			PetscReal a = gTz / pAp;

			if (tNorm < 1e-16) {
			//	PetscPrintf(PETSC_COMM_SELF, "BREAK \n");
				break;
			}

			VecAXPY(x, -a, p);
			VecAXPY(g, -a, temp);

			sPC->applyPC(g, z);

			gTzPrev = gTz;

			VecDot(g, z, &gTz);

			PetscReal beta = gTz / gTzPrev;
			VecAYPX(p, beta, z);

			VecNorm(g, NORM_2, &rNorm);

			if (gTz != gTz) {
				PetscPrintf(PETSC_COMM_WORLD, "ERROR!!! %a \n", rNorm);
				break;
			}
			MyLogger::Instance()->getTimer("Iteration")->stopTimer();
		}
	} else {
		VecSet(x, 0);
	}
}

ReCGSolver::~ReCGSolver() {
	clearSubspace();
	VecDestroy(p);
	VecDestroy(Ap);
}

void ReCGSolver::initSolver() {
	VecDuplicate(b, &Ap);
	VecDuplicate(g, &p);
	VecCopy(z, p);

	maxSize = 150;
	gCounter = 0;

	itManager.setTitle("ReCG");
}

void ReCGSolver::project() {

	if (P.size() > 0) {
		//PetscPrintf(PETSC_COMM_WORLD, "RE - projection\n");

		for (int i = 0; i < P.size(); i++) {

			PetscReal a;
			VecDot(P[i], g, &a);
			VecAXPY(x, -a / PAP[i], P[i]);
			VecAXPY(g, -a / PAP[i], AP[i]);
		}

		Vec temp;
		VecDuplicate(b, &temp);

		VecCopy(b, g);

		sApp->setRequiredPrecision(MAXPREC);
		sApp->applyMult(x, temp, &itManager);
		VecAYPX(g, -1, temp); // g = Ax - b

		VecDestroy(temp);

		sPC->applyPC(g, z);
		VecNorm(g, NORM_2, &rNorm);
		VecCopy(z, p);
	}
}

void ReCGSolver::clearSubspace() {
	for (int i = 0; i < P.size(); i++) {
		VecDestroy(P[i]);
		VecDestroy(AP[i]);
	}
	P.clear();
	AP.clear();
	PAP.clear();
}

void ReCGSolver::solve() {

	project();

	PetscReal relativeCoef = MAX(bNorm, r0Norm);
	PetscReal gTzPrev, gTz;

	//solProj->applyProjection(g);
	//	sPC->applyPC(g, z);
	VecDot(g, z, &gTz);

	while (!sCtr->isConverged(getItCount(), rNorm, relativeCoef, &g)) {

		itManager.setIterationData("Rel. err", rNorm / relativeCoef);
		nextIteration();

		PetscScalar pAp;
		sApp->applyMult(p, Ap, &itManager);
		VecDot(p, Ap, &pAp);

		PetscReal a = gTz / pAp;

		VecAXPY(x, -a, p);
		VecAXPY(g, -a, Ap);

		//Save for reortogonalization or projection


		//if (gCounter > maxSize) {

		//	VecCopy(p, P[gCounter % maxSize]);
		//	VecCopy(Ap, AP[gCounter % maxSize]);
		//	PAP[gCounter % maxSize] = pAp;
		//} else {
		Vec pT, ApT;
		VecDuplicate(p, &pT);
		VecCopy(p, pT);
		VecDuplicate(Ap, &ApT);
		VecCopy(Ap, ApT);

		P.push_back(pT);
		AP.push_back(ApT);
		PAP.push_back(pAp);
		//}
		gCounter++;

		sPC->applyPC(g, z);

		gTzPrev = gTz;

		VecDot(g, z, &gTz);

		PetscReal beta = gTz / gTzPrev;
		VecAYPX(p, beta, z);

		VecNorm(g, NORM_2, &rNorm);

		if (gTz != gTz) {
			PetscPrintf(PETSC_COMM_WORLD, "ERROR!!!");
			break;
		}

	}

}

void Solver::reset(Vec newB, Vec newX) {

	x = newX;
	b = newB;

	VecCopy(b, g);

	Vec temp;
	VecDuplicate(b, &temp);
	sApp->setRequiredPrecision(MAXPREC);
	sApp->applyMult(x, temp, &itManager);
	VecAYPX(g, -1, temp);
	VecDestroy(temp);

	sPC->applyPC(g, z);

	VecDot(g, z, &rNorm);
	rNorm = sqrt(rNorm);

	r0Norm = rNorm;

	itManager.reset();
}

void GLanczos::solve() {
	PetscInt n, locN, low, high;

	PetscReal *tempArr;
	VecGetSize(b, &n);
	VecGetLocalSize(b, &locN);
	VecGetOwnershipRange(b, &low, &high);

	Vec p;
	VecDuplicate(b, &p);

	VecScale(g, -1);

	if (prevSize > 0) {
		Vec v;
		VecCreateSeq(PETSC_COMM_SELF, prevSize, &v);
		MatMultTranspose(Vprev, g, v);

		PetscReal *vArr;
		PetscReal *tArr = new PetscReal[prevSize];

		VecGetArray(v, &vArr);

		tArr[0] = vArr[0];
		for (int i = 1; i < prevSize; i++) {
			tArr[i] = vArr[i] - lambda[i] * tArr[i - 1];
			//	PetscPrintf(comm, "tArr[%d] %f \n", i-1, tArr[i-1]);
		}

		vArr[prevSize - 1] = tArr[prevSize - 1] / mju[prevSize - 1];
		for (int i = prevSize - 2; i >= 0; i--) {
			vArr[i] = (tArr[i] - beta[i + 1] * vArr[i + 1]) / mju[i];
			//PetscPrintf(comm, "%f %f %f\n", tArr[i], beta[i], mju[i]);
		}

		VecRestoreArray(v, &vArr);

		PetscViewer vv;
		PetscViewerBinaryOpen(PETSC_COMM_WORLD, "../matlab/y.m", FILE_MODE_WRITE, &vv);
		VecView(v, vv);
		PetscViewerDestroy(vv);

		MatMult(Vprev, v, p);
		VecAXPY(x, 1, p);

		sApp->applyMult(x, g, &itManager);
		VecAXPY(g, -1, b);

		PetscReal normProj;
		VecNorm(g, NORM_2, &normProj);
		PetscPrintf(comm, "********************************\n  %f \n **********************\n", normProj);

		rNorm = normProj;

		if (sCtr->isConverged(itManager.getItCount(), rNorm, r0Norm, &g)) {
			PetscPrintf(comm, "Cinverged in the projection phase!!! (BRAVO) \n");
			return;
		}

		delete[] lambda, mju, beta;
	}

	Vec vNew, vCur, vPrev;
	VecDuplicate(b, &vNew);
	VecDuplicate(b, &vCur);
	VecDuplicate(b, &vPrev);

	VecSet(p, 0);

	PetscInt *lVecInd = new PetscInt[locN];
	for (int i = 0; i < locN; i++) {
		lVecInd[i] = i + low;
	}

	Mat V;

	MatCreateMPIDense(comm, locN, PETSC_DECIDE, PETSC_DECIDE, MAXSTEPS, PETSC_NULL, &V);
	PetscReal alpha[MAXSTEPS], phi[MAXSTEPS];

	lambda = new PetscReal[MAXSTEPS];
	mju = new PetscReal[MAXSTEPS];
	beta = new PetscReal[MAXSTEPS];

	VecCopy(g, vCur);
	VecCopy(g, vNew);

	VecNorm(vCur, NORM_2, beta);
	VecScale(vCur, 1 / beta[0]);
	lambda[0] = 0;
	phi[0] = beta[0];

	int k = 0;
	rNorm = fabs(phi[k]);

	VecGetArray(vCur, &tempArr);
	MatSetValues(V, locN, lVecInd, 1, &k, tempArr, INSERT_VALUES);
	MatAssemblyBegin(V, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(V, MAT_FINAL_ASSEMBLY);
	VecRestoreArray(vCur, &tempArr);

	while (!sCtr->isConverged(itManager.getItCount(), rNorm, r0Norm, &vNew)) {

		sApp->applyMult(vCur, vNew, &itManager);

		if (k > 0) {
			VecAXPY(vNew, -beta[k], vPrev);
		}

		VecDot(vNew, vCur, alpha + k);

		if (k > 0) {
			lambda[k] = beta[k] / mju[k - 1];
			phi[k] = -lambda[k] * phi[k - 1];
		}
		mju[k] = alpha[k] - lambda[k] * beta[k];

		VecAYPX(p, -beta[k], vCur);
		VecScale(p, 1 / mju[k]);
		VecAXPY(x, phi[k], p);

		VecAXPY(vNew, -alpha[k], vCur);

		VecNorm(vNew, NORM_2, beta + k + 1);
		if (beta[k + 1] < 1e-7) {
			k++; // < to maintain the correct step number count for extracting V
			break;
		}
		VecScale(vNew, 1 / beta[k + 1]);

		rNorm = fabs(phi[k]); //Absolute value of phi is equal to residual norm
		k++;

		//Insert new vector to the V matrix as a column
		VecGetArray(vNew, &tempArr);
		MatSetValues(V, locN, lVecInd, 1, &k, tempArr, INSERT_VALUES);
		MatAssemblyBegin(V, MAT_FINAL_ASSEMBLY);
		MatAssemblyEnd(V, MAT_FINAL_ASSEMBLY);
		VecRestoreArray(vNew, &tempArr);

		//Proceed to the next step
		VecCopy(vCur, vPrev);
		VecCopy(vNew, vCur);

		VecCopy(p, vNew);
		VecScale(vNew, phi[k - 1]); // For error estimation

		nextIteration();

		if (k == MAXSTEPS) break;
	}

	if (k > 1) {
		IS IScol, ISrow;
		ISCreateStride(PETSC_COMM_SELF, k - 1, 0, 1, &IScol);
		ISCreateStride(comm, locN, low, 1, &ISrow);
		MatGetSubMatrix(V, ISrow, IScol, MAT_INITIAL_MATRIX, &Vprev);

		prevSize = k - 1;

		ISDestroy(ISrow);
		ISDestroy(IScol);
	}

	MatDestroy(V);
	VecDestroy(vNew);
	VecDestroy(vCur);
	VecDestroy(vPrev);
	VecDestroy(p);
	delete[] lVecInd;
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

	PetscReal xNorm;
	VecNorm(x, NORM_2, &xNorm);

	while (!sCtr->isConverged(getItCount(), rNorm, bNorm, &g)) {

		outNorm = outNorm * 0.66;

		PetscReal reqPrec = fmax(fmin(outNorm, rNorm * 1e-1), 1e-6);
		sApp->setRequiredPrecision(reqPrec);
		if (reqPrec < pow((float) 10, (int) -1 * (gradRestartLoop))) {
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
		VecNorm(x, NORM_2, &xNorm);
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

