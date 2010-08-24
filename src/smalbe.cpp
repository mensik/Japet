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

	VecDuplicate(c, &tempMSize);
	VecDuplicate(c, &lmb);
	VecDuplicate(b, &temp);

}

Smalbe::~Smalbe() {
	VecDestroy(tempMSize);
	VecDestroy(temp);
	VecDestroy(x);
	VecDestroy(lmb);
}

void Smalbe::applyMult(Vec in, Vec out) {
	MatMult(B, in, tempMSize);
	MatMultTranspose(B, tempMSize, temp);
	MatMult(A, in, out);

	VecAXPY(out, ro, temp);
}

bool Smalbe::isConverged(PetscInt itNum, PetscScalar gpNorm, Vec *x) {
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
	PetscReal ANorm;
	MatNorm(A, NORM_1, &ANorm);

	PetscReal normBx = 1, normB;
	VecNorm(b, NORM_2, &normB);
	int outerIterations = 0;
	int innnerIterations = 0;
	while (normBx / normB > 1e-3) {
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

		PetscPrintf(PETSC_COMM_WORLD, "|Bx|/|b| = %f \t\tL = %f\n", normBx/normB, actualL);

		previousL = actualL;
	}

	PetscPrintf(PETSC_COMM_WORLD, "\nCompleted\n\nTotal inner iterations: %d\n\n", innnerIterations);
}

PetscReal Smalbe::Lagrangian() {
	PetscReal L = 0;

	PetscReal xAx;
	MatMultAdd(A, x, temp);
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
