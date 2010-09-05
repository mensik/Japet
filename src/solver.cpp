#include "solver.h"

Solver::Solver(Mat A, Vec b, Vec x) {
	this->A = A;
	this->b = b;
	this->x = x;

	sApp = this;

	init();
}

Solver::Solver(SolverApp *sa, Vec b, Vec x) {
	this->sApp = sa;
	this->b = b;
	this->x = x;
	init();
}

Solver::~Solver() {

	VecDestroy(g);
}

void Solver::init() {

	sCtr = this;
	itCounter = 0;
	isVerbose = false;
	precision = 1e-3;

	VecDuplicate(b, &g);

	Vec temp;
	VecDuplicate(b, &temp);

	VecCopy(b, g);

	sApp->applyMult(x, temp);
	VecAYPX(g, -1, temp);

	VecDestroy(temp);

	VecNorm(g, NORM_2, &rNorm);
}

void Solver::applyMult(Vec in, Vec out) {
	MatMult(A, in, out);
}

bool Solver::isConverged(PetscInt itNumber, PetscReal rNorm, Vec *r) {
	return rNorm < precision;
}

void Solver::setIterationData(std::string name, PetscReal value) {
	iterationData[name] = value;
}

void Solver::nextIteration() {
	IterationInfo info;
	info.itNumber = itCounter;
	info.rNorm = rNorm;

	if (isVerbose) PetscPrintf(PETSC_COMM_WORLD, "%d: |g|_2 = %f", itCounter, rNorm);

	for (std::map<std::string, PetscReal>::iterator i = iterationData.begin(); i
			!= iterationData.end(); i++) {
		const PetscReal data = i->second;
		info.itData.push_back(data);
		if (isVerbose) PetscPrintf(PETSC_COMM_WORLD, "\t%s=%f", i->first.c_str(), i->second);
	}
	if (isVerbose) PetscPrintf(PETSC_COMM_WORLD, "\n");

	itInfo.push_back(info);
	itCounter++;
}

void Solver::saveIterationInfo(const char *filename) {
	PetscInt rank;
	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

	if (!rank) {
		FILE *f;
		f = fopen(filename, "w");
		if (f != NULL) {
			PetscReal firstRNorm = itInfo[0].rNorm;

			fprintf(f, "#	itNumber\t|g|_2\tR");
			for (std::map<std::string, PetscReal>::iterator d = iterationData.begin(); d
					!= iterationData.end(); d++) {
				fprintf(f, "\t%s", d->first.c_str());
			}
			fprintf(f, "\n");

			for (std::vector<IterationInfo>::iterator i = itInfo.begin(); i
					!= itInfo.end(); i++) {

				fprintf(f, "%6d %11.4e %11.4e", i->itNumber, i->rNorm, std::pow((float)i->rNorm / firstRNorm,(float) 2 / (i->itNumber + 1)));
				for (std::vector<PetscReal>::iterator d = i->itData.begin(); d
						!= i->itData.end(); d++) {
					fprintf(f, "%11.4e", *d);
				}
				fprintf(f, "\n");
			}
			fclose(f);
		}
	}
}

void CGSolver::initSolver() {
	VecDuplicate(b, &temp);
	VecDuplicate(g, &p);
	VecCopy(g, p);

}

CGSolver::~CGSolver() {
	VecDestroy(p);
	VecDestroy(temp);
}

void CGSolver::solve() {

	while (!sCtr->isConverged(getItCount(), rNorm, &g)) {

		nextIteration();

		PetscScalar pAp;
		sApp->applyMult(p, temp);
		VecDot(p, temp, &pAp);

		PetscReal a = (rNorm * rNorm) / pAp;
		VecAXPY(x, -a, p);
		VecAXPY(g, -a, temp);

		PetscReal rNormS;
		VecNorm(g, NORM_2, &rNormS);

		PetscReal beta = (rNormS * rNormS) / (rNorm * rNorm);
		VecAYPX(p, beta, g);

		rNorm = rNormS;

	}
}

void ASinStep::initSolver() {
	tau = 1e-8;
	fi = (sqrt(5) - 1) / 2;
	z = 0;
}

void ASinStep::solve() {
	PetscReal beta;
	Vec Ag;
	VecDuplicate(g, &Ag);

	for (int i = 0; i < 2; i++) {

		sApp->applyMult(g, Ag);

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

		nextIteration();
		VecNorm(g, &rNorm);
	}
	while (!sCtr->isConverged(getItCount(), rNorm, &g)) {
		sApp->applyMult(g, Ag);

		PetscReal gg, gAg, moment1;
		VecDot(g, g, &gg);
		VecDot(Ag, g, &gAg);
		moment1 = gAg / gg;
		m = fmin(m, moment1);
		M = fmax(M, moment1);

		if (getItCount() % 2 == 0) {
			PetscReal eps = tau * (M - m);
			z += fi;
			beta = m + eps + (std::cos(PI * z) + 1) * (M - m - 2 * eps) / 2;
		} else {
			beta = M + m - beta;
		}

		VecAXPY(x, -1 / beta, g);
		VecAXPY(g, -1 / beta, Ag);

		setIterationData("m", m);
		setIterationData("M", M);
		setIterationData("beta", beta);

		nextIteration();
		VecNorm(g, &rNorm);
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

