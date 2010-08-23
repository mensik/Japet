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

void Smalbe::solve() {
	PetscPrintf(PETSC_COMM_WORLD, "Here goes Smalbe!! \n");

	Vec bCopy;
	VecDuplicate(b, &bCopy);
	VecCopy(b, bCopy);

	PetscReal ANorm;
	MatNorm(A, NORM_1, &ANorm);


	MPRGP *mprgp = new MPRGP(this, b, L, x, 1, 2 / (ANorm + ro));
	mprgp->solve();
	delete mprgp;

	MatMult(B, x, tempMSize);
	VecAXPY(lmb, ro, tempMSize);

	ro = ro * 2;

	mprgp = new MPRGP(this, b, L, x, 1, 2 / (ANorm));
	mprgp->solve();

	PetscReal bbb;
	VecAXPY(bCopy, -1,b);
	VecNorm(bCopy, NORM_1, &bbb);
	PetscPrintf(PETSC_COMM_WORLD, "%f\n", bbb);

	delete mprgp;
}

void Smalbe::dumpSolution(PetscViewer v) {
	VecView(x, v);
}
