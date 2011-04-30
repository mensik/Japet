#include "smale.h"

SDSystem::SDSystem(Mesh *mesh, PetscReal (*f)(Point), PetscReal (*K)(Point)) {
	Mat Agl;
	Vec bgl;
	Vec lmb;
	FEMAssemble2DLaplace(PETSC_COMM_WORLD, mesh,Agl,bgl,f,K);
	GenerateJumpOperator(mesh,B,lmb);
	extractLocalAPart(Agl, &A);

	PetscInt mA,nA, mB, nB;
	MatGetSize(A, &mA, &nA);
	MatGetSize(B, &mB, &nB);
	
	VecCreateSeq(PETSC_COMM_SELF, mA, &b);
	VecCreateMPI(PETSC_COMM_WORLD,PETSC_DECIDE,  mB, &c);
	VecSet(c, 0);

	Vec tempG, tempLoc;
	VecCreateGhost(PETSC_COMM_WORLD, mA, PETSC_DECIDE, 0, PETSC_NULL,  &tempG);
	VecCopy(bgl, tempG);
	VecGhostGetLocalForm(tempG, &tempLoc);
	VecCopy(tempLoc, b);

	VecDestroy(tempLoc);
	VecDestroy(tempG);

	MatDestroy(Agl);
	VecDestroy(bgl);
	VecDestroy(lmb);
}

Smale::Smale(SDSystem *sd, PetscReal mi, PetscReal ro, PetscReal beta, PetscReal M) {
	this->sd = sd;
	this->mi = mi;
	this->ro = ro;
	this->beta = beta;
	this->M = M;

	outItCount = 0;

	PetscInt m,n;
	MatGetSize(sd->getA(), &m, &n);
	
	VecCreateGhost(PETSC_COMM_WORLD, n, PETSC_DECIDE, 0, PETSC_NULL, &x);
	VecCreateGhost(PETSC_COMM_WORLD, n, PETSC_DECIDE, 0, PETSC_NULL, &g);
	VecCreateGhost(PETSC_COMM_WORLD, n, PETSC_DECIDE, 0, PETSC_NULL, &p);
	VecCreateGhost(PETSC_COMM_WORLD, n, PETSC_DECIDE, 0, PETSC_NULL, &temp);

	VecGetSize(sd->getc(), &m);
	VecCreate(PETSC_COMM_WORLD, &l);
	VecSetSizes(l, PETSC_DECIDE, m);
	VecSetFromOptions(l);

	PetscInt gn;
	VecGetSize(x, &gn);

	VecGhostGetLocalForm(x, &lx);
	VecGhostGetLocalForm(g, &lg);
	VecGhostGetLocalForm(p, &lp);
	VecGhostGetLocalForm(temp, &ltemp);

	VecCreate(PETSC_COMM_WORLD, &tempMSize);
	VecSetSizes(tempMSize, PETSC_DECIDE, m);
	VecSetFromOptions(tempMSize);
	VecDuplicate(tempMSize,&bxc);

	lPrec = 1e-5;
	prevL = -1;
	aL = -1;

	VecSet(x,0);
	VecSet(l,0);
}

Smale::~Smale() {
	VecDestroy(x);
	VecDestroy(l);


	VecDestroy(temp);
	VecDestroy(ltemp);	
	VecDestroy(tempMSize);
	VecDestroy(bxc);
	VecDestroy(g);
	VecDestroy(p);
	VecDestroy(lx);
	VecDestroy(lg);
	VecDestroy(lp);
}

void Smale::solve() { 
	VecScale(sd->getb(), -1);
	VecScale(sd->getc(), -1);
	
	MatMultAdd(sd->getB(), x, sd->getc(), bxc);
	refreshGradient();
	VecNorm(g,NORM_2, &gNorm);

	while (!isOuterConverged()) {

		inItCount[outItCount] = 0;
		VecCopy(g,p);
		prevL = aL;	
		while (!isInerConverged()) {
			inItCount[outItCount]++;
			
			MatMult(sd->getB(), p, tempMSize);
			MatMultTranspose(sd->getB(), tempMSize, temp);
			VecScale(temp, ro);
			
			MatMultAdd(sd->getA(), lp,ltemp, ltemp);	
			
			PetscReal pAp;
			VecDot(p, temp, &pAp);
			PetscReal a = (gNorm*gNorm) / pAp;
			
			VecAXPY(x, -a, p);
			VecAXPY(g, -a, temp);

			PetscReal gNormNew;
			VecNorm(g, NORM_2, &gNormNew);
			PetscReal b = (gNormNew * gNormNew)/(gNorm*gNorm);

			VecAYPX(p,b,g);
			gNorm = gNormNew;


			
			MatMultAdd(sd->getB(), x, sd->getc(), bxc);
		}
		
		updateLagrange();		
		aL = L();

		if (prevL > 0) { // It is -1 at first iteration
			PetscReal normBxc;
			VecNorm(bxc, NORM_2, &normBxc);
			if (aL < (prevL + 0.5 * normBxc*normBxc)) {
				ro = ro*beta;
				PetscPrintf(PETSC_COMM_WORLD, "Now, ro = %e\n", ro);
			}
		}
	
		PetscPrintf(PETSC_COMM_WORLD, "In it count:\t%d\tgNorm:\t%e\tLagrangian:\t%e\n", inItCount[outItCount],gNorm, aL);

		refreshGradient();
		VecNorm(g,NORM_2, &gNorm);
		outItCount++;
	}
}

bool Smale::isInerConverged() {
	PetscReal bxNorm;

	VecNorm(bxc, NORM_2, &bxNorm);
	
	PetscReal normBound = M*bxNorm < mi ? M*bxNorm : mi;

	return gNorm < normBound;
}

bool Smale::isOuterConverged() {
	return (gNorm < 1e-5);
}

PetscReal Smale::L() {
	PetscReal L = 0;
	
	PetscReal xAx;
	MatMult(sd->getA(), lx,ltemp);
	VecDot(x,temp,&xAx);
	
	PetscReal lbx;
	PetscReal bx=0;
	VecDot(sd->getb(), lx, &lbx);	
	MPI_Allreduce(&lbx, &bx, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	
	PetscReal bxcl;
	VecDot(bxc, l, &bxcl);

	PetscReal normBxc;
	VecNorm(bxc, NORM_2, &normBxc);
	
	L = 0.5 * xAx - bx + bxcl + 0.5*ro*normBxc*normBxc;

	return L;	
}

void Smale::refreshGradient() {
	VecWAXPY(tempMSize, ro, bxc, l);
	MatMultTranspose(sd->getB(), tempMSize, g);	
	
	MatMultTransposeAdd(sd->getA(),lx,sd->getb(),ltemp);
	VecAXPY(lg, 1,ltemp);
}

void Smale::updateLagrange() {
	// l = l + ro(Bx - c)	
	VecAXPY(l, ro, bxc);
}

void Smale::dump(PetscViewer v) {
	PetscObjectSetName((PetscObject)x,"u");
	VecView(x,v);
	PetscObjectSetName((PetscObject)l,"l");
	VecView(l,v);
	PetscObjectSetName((PetscObject)sd->getB(),"B");
	MatView(sd->getB(),v);
}

void Smale::dumpSolution(PetscViewer v) {
	PetscObjectSetName((PetscObject)x,"u");
	VecView(x,v);
}
