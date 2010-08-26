#include "smalbe.h"

Smalbe::Smalbe(Mat A, Vec b, Mat B, Vec c, Vec L, PetscReal mi, PetscReal ro,
		PetscReal beta, PetscReal M) {
	this->A = A;
	this->B = B;
	this->b = b;
	this->c = c;
	this->L = L;
	this->mi = mi;
	this->ro = ro;
	this->beta = beta;
	this->M = M;

	VecDuplicate(b, &x);
	extractLocalAPart(A, &(this->A));

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
	VecDestroy(x);
	VecDestroy(lmb);
}

void Smalbe::applyMult(Vec in, Vec out) {
	MatMult(B, in, tempMSize);
	MatMultTranspose(B, tempMSize, out);
	VecCopy(in, temp);
	MatMult(A, tempGh, temp2Gh);

	VecAYPX(out, ro, temp2);
}

bool Smalbe::isConverged(PetscInt itNum, PetscReal gpNorm, Vec *x) {
	MatMult(B, *x, tempMSize);
	PetscReal normBx;
	VecNorm(tempMSize, &normBx);
	PetscReal conv = fmin(normBx * M, mi);
	return gpNorm < conv;
}

void Smalbe::solve() {
	PetscPrintf(PETSC_COMM_WORLD, "Here goes Smalbe!! \n");

	Vec bCopy;
	VecDuplicate(b, &bCopy);
	VecCopy(b, bCopy);

	PetscReal actualL = 0, previousL = -1;
	PetscReal ANorm, ANormLoc;

	MatNorm(A, NORM_1, &ANormLoc);
	MPI_Allreduce(&ANormLoc, &ANorm, 1, MPI_DOUBLE, MPI_MAX, PETSC_COMM_WORLD);

	PetscReal normBx = 1;
	int outerIterations = 0;
	int innnerIterations = 0;
	while (normBx > 1e-3) {
		outerIterations++;

		MatMultTranspose(B, lmb, bCopy);
		VecAYPX(bCopy, -1, b);

		MPRGP *mprgp = new MPRGP(this, bCopy, L, x, 1, 2 / (ANorm + ro));
		mprgp->setCtrl(this);
		mprgp->solve();
		int ii = mprgp->getNumIterations();
		PetscPrintf(PETSC_COMM_WORLD, "\n%d. Inner iterations: %d\n", outerIterations, ii);
		innnerIterations += ii;

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



		if (outerIterations > 1) {

			VecNorm(tempMSize, NORM_2, &normBx);
			if (actualL < (previousL + 0.5 * normBx * normBx)) {
				ro = ro * beta;
				PetscPrintf(PETSC_COMM_WORLD, "ro = %e\n", ro);
			}
		}
		VecNorm(tempMSize, NORM_1, &normBx);
		PetscPrintf(PETSC_COMM_WORLD, "|Bx|_max = %f \t\tL = %f\n", normBx, actualL);

		previousL = actualL;
	}

	PetscPrintf(PETSC_COMM_WORLD, "\nCompleted\n\nTotal inner iterations: %d\n\n", innnerIterations);
}

PetscReal Smalbe::Lagrangian() {
	PetscReal L = 0;

	PetscReal xAx;
	VecCopy(x, temp2);
	MatMultAdd(A, temp2Gh, tempGh);
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
