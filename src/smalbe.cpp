#include "smalbe.h"

Smalbe::Smalbe(Mat A, Vec b, Mat B, Vec c, Vec L, PetscReal h, PetscReal mi, PetscReal ro,
		PetscReal beta, PetscReal M) {
	//this->A = A;
	this->B = B;
	this->b = b;
	this->c = c;
	this->L = L;
	this->mi = mi;
	this->ro = ro;
	this->beta = beta;
	this->M = M;
	this->h = h;
	logFileName = "Smalbe_default";
	title = "Smalbe";

	VecDuplicate(b, &x);
	extractLocalAPart(A, &(this->A));
	getLocalJumpPart(B, &Bloc);

	MatDuplicate(this->A, MAT_DO_NOT_COPY_VALUES, &AroLoc);

	initPC();

	PetscInt nA;
	MatGetSize(this->A, &nA, PETSC_NULL);

	VecCreateGhost(PETSC_COMM_WORLD, nA, PETSC_DECIDE, 0, PETSC_NULL, &temp);
	VecCreateGhost(PETSC_COMM_WORLD, nA, PETSC_DECIDE, 0, PETSC_NULL, &temp2);
	VecGhostGetLocalForm(temp, &tempGh);
	VecGhostGetLocalForm(temp2, &temp2Gh);
	VecDuplicate(c, &tempMSize);
	VecDuplicate(c, &lmb);

}

Smalbe::~Smalbe() {
	VecDestroy(tempMSize);
	VecDestroy(tempGh);
	VecDestroy(temp2Gh);
	VecDestroy(temp);
	VecDestroy(temp2);

	MatDestroy(A);
	MatDestroy(AroLoc);
	MatDestroy(Bloc);

	VecDestroy(x);
	VecDestroy(lmb);
	PCDestroy(pc);
}

void Smalbe::initPC() {
	PCCreate(PETSC_COMM_SELF, &pc);

	MatMatMultTranspose(Bloc, Bloc, MAT_REUSE_MATRIX, 1, &AroLoc);
	MatAYPX(AroLoc, ro, A, DIFFERENT_NONZERO_PATTERN);

	PCSetOperators(pc, A, A, SAME_PRECONDITIONER);
	PCSetFromOptions(pc);

//
//	PCSetType(pc,PCMG);
//	MPI_Comm comm = PETSC_COMM_SELF;
//	PCMGSetLevels(pc,1, &comm);
//	PCMGSetType(pc,PC_MG_MULTIPLICATIVE);

	PCSetUp(pc);
}

void Smalbe::applyMult(Vec in, Vec out) {
	MatMult(B, in, tempMSize);
	MatMultTranspose(B, tempMSize, out);
	VecCopy(in, temp);
	MatMult(A, tempGh, temp2Gh);

	VecAYPX(out, ro, temp2);
}

bool Smalbe::isConverged(PetscInt itNum, PetscReal gpNorm, PetscReal bNorm, Vec *x) {
	MatMult(B, *x, tempMSize);
	PetscReal normBx;
	//VecNorm(tempMSize, NORM_2, &normBx);
 VecNorm(tempMSize, NORM_MAX, &normBx);
	PetscReal conv = normBx * M;
	if (conv > mi) conv = mi;
	return (gpNorm < conv);
}

void Smalbe::applyPC(Vec g, Vec z) {
	VecCopy(g, temp);
	PCApply(pc, tempGh, temp2Gh);
	VecCopy(temp2, z);
}

void Smalbe::solve() {
	PetscPrintf(PETSC_COMM_WORLD, "Here goes Smalbe!! \n");

	remove((logFileName + IN_ITERATION_SUFFIX).c_str());
	itManager.setTitle(title);
	Vec bCopy;
	VecDuplicate(b, &bCopy);
	VecCopy(b, bCopy);

	PetscReal actualL = 0, previousL = -1;
	PetscReal ANorm, ANormLoc;

	MatNorm(A, NORM_1, &ANormLoc);
	MPI_Allreduce(&ANormLoc, &ANorm, 1, MPI_DOUBLE, MPI_MAX, PETSC_COMM_WORLD);

	PetscReal normBx = 1;
	int innnerIterations = 0;

	while (normBx > 1e-3) {

		MatMultTranspose(B, lmb, bCopy);
		VecAYPX(bCopy, -1, b);

		MPRGP *mprgp = new MPRGP(this, bCopy, L, x, 1, 2 / (ANorm + ro));

		mprgp->setSolverCtr(this);
		mprgp->setPC(this);

		std::string title("MPRGP cycle no.");
		std::stringstream oss;
		oss << title << ' ' << itManager.getItCount();
		mprgp->setTitle(oss.str());

		mprgp->solve();
		mprgp->saveIterationInfo((logFileName + IN_ITERATION_SUFFIX).c_str(),false);

		innnerIterations += mprgp->getItCount();
		itManager.setIterationData("3. Inner it.count", mprgp->getItCount());
		itManager.setIterationData("2. (z*g)^0.5", mprgp->getNormG());
		delete mprgp;

		MatMult(B, x, tempMSize);
		actualL = Lagrangian();
		VecAXPY(lmb, ro, tempMSize);

		//		Vec xTemp;
		//		VecDuplicate(x, &xTemp);
		//		MatMult(A, x, xTemp);
		//		VecAXPY(xTemp, -1, bCopy);
		//
		//		PetscReal norm;
		//		VecNorm(xTemp, &norm);
		//		PetscPrintf(PETSC_COMM_WORLD, "|Ax - b + B'lmb| = %f\n", norm);


		if (itManager.getItCount() > 0) {

			VecNorm(tempMSize, NORM_2, &normBx);
			if (actualL < (previousL + 0.5 * normBx * normBx)) {
				ro = ro * beta;
				PCDestroy(pc);
				initPC();
			}
		}
		VecNorm(tempMSize, NORM_MAX, &normBx);
	//	normBx *= sqrt(h);
		previousL = actualL;

		itManager.setIterationData("1. sqrt(h) |Bx|", normBx);
		itManager.setIterationData("5. L", actualL);
		itManager.setIterationData("4. ro",ro);

		itManager.nextIteration();
		itManager.saveIterationInfo((logFileName + OUT_ITERATION_SUFFIX).c_str());
	}

	PetscPrintf(PETSC_COMM_WORLD, "\nCompleted\n\nTotal inner iterations: %d\n\n", innnerIterations);
}

PetscReal Smalbe::Lagrangian() {
	PetscReal L = 0;

	PetscReal xAx;
	VecCopy(x, temp2);
	MatMult(A, temp2Gh, tempGh);
	VecDot(x, temp, &xAx);

	PetscReal bx;
	VecDot(b, x, &bx);

	PetscReal bxcl;
	VecDot(tempMSize, lmb, &bxcl);

	PetscReal normBx;
	VecNorm(tempMSize, NORM_2, &normBx);

	L = 0.5 * xAx - bx + bxcl + 0.5 * ro * normBx * normBx;

	return L;
}

void Smalbe::dumpSolution(PetscViewer v) {
	VecView(x, v);
}
