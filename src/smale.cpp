#include "smale.h"

SDRectSystem::SDRectSystem(PetscReal m, PetscReal n, PetscReal k, PetscReal l, PetscReal h, PetscInt xSize, PetscInt ySize, PetscScalar (*f)(Point), PetscScalar (*K)(Point)) {
	subMeshCount = xSize * ySize;
  MPI_Comm_rank(PETSC_COMM_WORLD,&localIndex);
	layout = new DomainRectLayout(xSize,ySize);

	PetscReal xWidth = (n - m) / xSize;
	PetscReal yWidth = (l - k) / ySize;
	
	subMesh = new RectMesh *[subMeshCount];

	//:TODO:
	/// @todo Neni treba, aby kazdy proces generoval vsechny site. Uplne by melo
	/// stacit, aby kazdy rozeslal indexy svych hranicnich prvku

	for (int i = 0; i < subMeshCount; i++) {
		PetscInt xCoords, yCoords;
		layout->getMyCoords(i, xCoords, yCoords);
		
		PetscReal xStart = m + xCoords*xWidth;
		PetscReal xEnd = m + (xCoords+1)*xWidth;
		PetscReal yStart = k + yCoords*yWidth;
		PetscReal yEnd = k + (yCoords+1)*yWidth;
		
		subMesh[i] = new RectMesh(xStart,xEnd,yStart,yEnd,h);
		//PetscPrintf(PETSC_COMM_SELF, "%d: Mesh no. %d generated \n", localIndex, i);
	}

	//ierr = PetscPrintf(PETSC_COMM_SELF, "[%d,%d]: <%e,%e> x <%e,%e>\n", xCoords, yCoords, xStart, xEnd, yStart,yEnd);CHKERRQ(ierr);
	
	FEMAssemble2D(PETSC_COMM_SELF, subMesh[localIndex],A,b, f, K);
	

	//Urceni rozmeru matice B a vektoru c
	//if (localIndex == 0) { // Only root's job	
		PetscInt totalNodesCount = 0;
		PetscInt nodeBoundsCount = 0;
		PetscInt startIndexes[subMeshCount];
		
		for (int i = 0; i < subMeshCount; i++) {
			PetscInt xCoords, yCoords;
			layout->getMyCoords(i, xCoords, yCoords);
			startIndexes[i] = totalNodesCount;
			totalNodesCount += subMesh[i]->numPoints;
			if (xCoords < xSize - 1) {
				nodeBoundsCount+=subMesh[i]->yPoints;	
				if (yCoords > 0) nodeBoundsCount--; //Vynechani nadbytecne vazby ve ctverci
			}
	
			if (yCoords < ySize - 1) {
				nodeBoundsCount+=subMesh[i]->xPoints;
			}
		}

		MatCreate(PETSC_COMM_WORLD,&B);
		MatSetType(B,MATAIJ);
		MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,nodeBoundsCount,totalNodesCount);

		VecCreate(PETSC_COMM_WORLD,&c);
		VecSetSizes(c, PETSC_DECIDE, nodeBoundsCount);	
		VecSetFromOptions(c);
		VecSet(c,0);
		
			
		PetscInt counter = 0;
		if (localIndex == 0) { //Only MASTER's work
		for (int i = 0; i < subMeshCount; i++) {
			PetscInt xCoords, yCoords;
			layout->getMyCoords(i, xCoords, yCoords);
			PetscInt SI = startIndexes[i];
			if (xCoords < xSize - 1) {
				PetscInt neibDomain = layout->getSub(xCoords+1, yCoords);
				PetscInt neibSI = startIndexes[neibDomain];
				for (int j = (yCoords > 0)?1:0; j < subMesh[i]->yPoints; j++) {
					MatSetValue(B,counter, subMesh[i]->iR[j] + SI,1,INSERT_VALUES);
					MatSetValue(B,counter++, subMesh[neibDomain]->iL[j] + neibSI,-1,INSERT_VALUES);
				}
			}
	
			if (yCoords < ySize - 1) {
				PetscInt neibDomain = layout->getSub(xCoords, yCoords+1);
				PetscInt neibSI = startIndexes[neibDomain];
				for (int j = 0; j < subMesh[i]->xPoints; j++) {
					MatSetValue(B,counter, subMesh[i]->iT[j] + SI,1,INSERT_VALUES);
					MatSetValue(B,counter++, subMesh[neibDomain]->iB[j] + neibSI,-1,INSERT_VALUES);
				}
			}
		}
		}	
		MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);
		MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);
	//}
		//MatView(B,PETSC_VIEWER_DRAW_WORLD);
}

SDRectSystem::~SDRectSystem() {
	for (int i = 0; i < subMeshCount; i++) {
		delete subMesh[i];
	}
	delete layout;
	MatDestroy(A);
	VecDestroy(b);
	//if (localIndex == 0) {
		MatDestroy(B);
		VecDestroy(c);
	//}
	delete [] subMesh;
}

void SDRectSystem::setDirchletBound(PetscInt n, BoundSide *sides) {
	PetscInt xCoords, yCoords;
	layout->getMyCoords(localIndex, xCoords, yCoords);
	
	for (int i = 0; i < n; i++) {
		BoundSide side = sides[i];
		if ((side == ALL || side == LEFT) && xCoords == 0) {
			FEMSetDirchletBound(A,b,subMesh[localIndex]->xPoints, subMesh[localIndex]->iL);
		}
		if ((side == ALL || side == RIGHT) && xCoords == layout->getXSize() - 1) {
			FEMSetDirchletBound(A,b,subMesh[localIndex]->yPoints, subMesh[localIndex]->iR);
		}
		if ((side == ALL || side == BOTTOM) && yCoords == 0) {
			FEMSetDirchletBound(A,b,subMesh[localIndex]->xPoints, subMesh[localIndex]->iB);
		}
		if ((side == ALL || side == TOP) && yCoords == layout->getYSize() - 1) {
			FEMSetDirchletBound(A,b,subMesh[localIndex]->xPoints, subMesh[localIndex]->iT);
		}
	}
}

Smale::Smale(SDRectSystem *sd, PetscReal mi, PetscReal ro, PetscReal beta, PetscReal M) {
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
			
			MatMultAdd(sd->getA(), lp,ltemp);	
			
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
	return (gNorm < lPrec);
}

PetscReal Smale::L() {
	PetscReal L = 0;
	
	PetscReal xAx;
	MatMultAdd(sd->getA(), lx,ltemp);
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


SassiRectSystem::SassiRectSystem(PetscReal m, PetscReal n, PetscReal k, PetscReal l, PetscReal h, PetscInt xSize, PetscInt ySize, PetscScalar (*f)(Point), PetscScalar (*K)(Point), PetscReal rr) {
  
	r = rr;

	MPI_Comm_rank(PETSC_COMM_WORLD,&localIndex);
	layout = new DomainRectLayout(xSize,ySize);

	PetscReal xWidth = (n - m) / xSize;
	PetscReal yWidth = (l - k)/ ySize;
	PetscInt subDomainsCount = xSize * ySize;


	PetscInt xCoords, yCoords;
	layout->getMyCoords(localIndex, xCoords, yCoords);
	
	PetscReal xStart = m + xCoords*xWidth;
	PetscReal xEnd = m + (xCoords+1)*xWidth;
	PetscReal yStart = k + yCoords*yWidth;
	PetscReal yEnd = k + (yCoords+1)*yWidth;
	
	mesh = new RectMesh(xStart,xEnd,yStart,yEnd,h);
	
	FEMAssemble2D(PETSC_COMM_SELF, mesh, A,b, f, K);

	VecCreateGhost(PETSC_COMM_WORLD, mesh->numPoints, PETSC_DECIDE,0,PETSC_NULL, &x);	
	VecGhostGetLocalForm(x,&x_loc);

	PetscInt subBorderCounter[subDomainsCount];
	//Spocita, kolik ma dana tato subdomena hranic a pripravi pole
	numB = 0;
	if (xCoords > 0)
		numB++;
	if (yCoords > 0)
		numB++;
	if (xCoords < xSize - 1)
		numB++;
	if (yCoords < ySize - 1) 
		numB++;
	B = new Mat[numB];
	lag = new Vec[numB];
	indQ = new PetscInt[numB];
	ghQ = new Vec[numB];
	//ghRes = new Vec[numB];
	
	//Pripravi pole hranic pro vsechny
	q = new Vec[(xSize - 1)*ySize + xSize*(ySize - 1)];
	borderRes = new Vec[(xSize - 1)*ySize + xSize*(ySize - 1)];

	for (int i = 0; i < subDomainsCount; i++) {
		subBorderCounter[i] = 0;	//pripravi pocitadla jiz pripravenych hran pro jednotlive podoblasti
	}


	numQ = 0;
	for (int i = 0; i < subDomainsCount; i++) {
		PetscInt x,y;
		layout->getMyCoords(i,x,y);
		if (x < xSize - 1) { // zjisti, zda je napravo jeste oblast - tedy nutnost pripravit hranu
			
			PetscInt neibSubIndex = layout->getSub(x+1,y);
			
			if (neibSubIndex == localIndex) {
				PetscInt allInd[mesh->yPoints];
				for (int j = 0; j < mesh->yPoints; j++)
					allInd[j]=j;
				VecCreateGhost(MPI_COMM_WORLD, 0, PETSC_DECIDE,mesh->yPoints, allInd, &q[numQ]);
				VecCreateGhost(MPI_COMM_WORLD, 0, PETSC_DECIDE,mesh->yPoints, allInd, &borderRes[numQ]);
			} else {
				VecCreateGhost(MPI_COMM_WORLD, (localIndex == i ? mesh->yPoints : 0),PETSC_DECIDE,0, PETSC_NULL, &q[numQ]);
				VecCreateGhost(MPI_COMM_WORLD, (localIndex == i ? mesh->yPoints : 0),PETSC_DECIDE,0, PETSC_NULL, &borderRes[numQ]);
			}
				
			VecSet(q[numQ], 0);
			
			if (i == localIndex) { // prida hranu, ptatri li teto oblasti
				prepareBorder(B[subBorderCounter[i]], lag[subBorderCounter[i]], mesh->numPoints, mesh->yPoints, mesh->iR);
				indQ[subBorderCounter[i]] = numQ;
			}
			subBorderCounter[i]++;

			if (neibSubIndex == localIndex) { // prida hranu, je li tato oblast sousedem
				prepareBorder(B[subBorderCounter[neibSubIndex]], lag[subBorderCounter[neibSubIndex]], mesh->numPoints, mesh->yPoints, mesh->iL);
				indQ[subBorderCounter[neibSubIndex]] = numQ;
			}
			subBorderCounter[neibSubIndex]++;
			numQ++;
		}
		if (y < ySize - 1)  {	
			
			PetscInt neibSubIndex = layout->getSub(x,y + 1);
			
			if (neibSubIndex == localIndex) {
				PetscInt allInd[mesh->xPoints];
				for (int j = 0; j < mesh->xPoints; j++)
					allInd[j]=j;
				VecCreateGhost(MPI_COMM_WORLD, 0, PETSC_DECIDE,mesh->xPoints, allInd, &q[numQ]);
				VecCreateGhost(MPI_COMM_WORLD, 0, PETSC_DECIDE,mesh->xPoints, allInd, &borderRes[numQ]);
			} else {
				VecCreateGhost(MPI_COMM_WORLD, (localIndex == i ? mesh->xPoints : 0),PETSC_DECIDE,0, PETSC_NULL, &q[numQ]);
				VecCreateGhost(MPI_COMM_WORLD, (localIndex == i ? mesh->xPoints : 0),PETSC_DECIDE,0, PETSC_NULL, &borderRes[numQ]);
			}

			VecSet(q[numQ], 0);
			
			if (i == localIndex) {
				prepareBorder(B[subBorderCounter[i]], lag[subBorderCounter[i]], mesh->numPoints, mesh->xPoints, mesh->iT);
				indQ[subBorderCounter[i]] = numQ;
			}
			subBorderCounter[i]++;
			if (neibSubIndex == localIndex) {
				prepareBorder(B[subBorderCounter[neibSubIndex]], lag[subBorderCounter[neibSubIndex]], mesh->numPoints, mesh->xPoints, mesh->iB);
				indQ[subBorderCounter[neibSubIndex]] = numQ;
			}
			subBorderCounter[neibSubIndex]++;
			numQ++;
		}
	}
}

SassiRectSystem::~SassiRectSystem() {
	MatDestroy(A);
	VecDestroy(b);
	VecDestroy(x);
	VecDestroy(x_loc);
	for (int i = 0; i < numB; i++) {
		MatDestroy(B[i]);
		VecDestroy(lag[i]);
		VecDestroy(ghQ[i]);
	//	VecDestroy(ghRes[i]);
	}
	for (int i = 0; i < numQ; i++) {
		VecDestroy(q[i]);
		VecDestroy(borderRes[i]);
	}

	delete [] ghQ;
	delete [] B;
	delete [] lag;
	delete [] indQ;
	delete [] q;
	delete [] borderRes;
	//delete [] ghRes;

	delete mesh;
	delete layout;
}

void SassiRectSystem::prepareBorder(Mat &B, Vec &lag, PetscInt numNodes, PetscInt borderLenght, PetscInt *indexes) {
	MatCreateSeqAIJ(PETSC_COMM_SELF, borderLenght,numNodes, 1, PETSC_NULL, &B);

	for (int i = 0; i < borderLenght; i++) {
		MatSetValue(B, i, indexes[i], 1, INSERT_VALUES);
	}
	MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);
	VecCreateSeq(PETSC_COMM_SELF, borderLenght, &lag);
	VecSet(lag, 0);
	MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);
}

void SassiRectSystem::setDirchletBound(PetscInt n, BoundSide *sides) {
	PetscInt xCoords, yCoords;
	layout->getMyCoords(localIndex, xCoords, yCoords);
	
	for (int i = 0; i < n; i++) {
		BoundSide side = sides[i];
		if ((side == ALL || side == LEFT) && xCoords == 0) {
			FEMSetDirchletBound(A,b,mesh->xPoints, mesh->iL);
		}
		if ((side == ALL || side == RIGHT) && xCoords == layout->getXSize() - 1) {
			FEMSetDirchletBound(A,b,mesh->yPoints, mesh->iR);
		}
		if ((side == ALL || side == BOTTOM) && yCoords == 0) {
			FEMSetDirchletBound(A,b,mesh->xPoints, mesh->iB);
		}
		if ((side == ALL || side == TOP) && yCoords == layout->getYSize() - 1) {
			FEMSetDirchletBound(A,b,mesh->xPoints, mesh->iT);
		}
	}
}

void SassiRectSystem::solve() {

		Mat Ar;
		Vec br;

		Mat tempMat[numB+1];

		Vec ghBorderRes[numB];

		for (int i = 0; i < numQ; i++) {
			VecGhostUpdateBegin(q[i], INSERT_VALUES, SCATTER_FORWARD);
			VecGhostUpdateEnd(q[i], INSERT_VALUES, SCATTER_FORWARD);
		}

		for (int i = 0; i < numB; i++) {
			MatCreateSeqAIJ(mesh->numPoints, mesh->numPoints,&tempMat[i]);
			MatMatMultTranspose(B[i],B[i],MAT_INITIAL_MATRIX,PETSC_DEFAULT,&tempMat[i]);
			MatScale(tempMat[i], r);
			VecGhostGetLocalForm(q[indQ[i]], &ghQ[i]);
		}
		tempMat[numB] = A;	
		MatCreateComposite(PETSC_COMM_SELF, numB + 1, tempMat, &Ar);

		int j = 0;
		PetscReal residualNorm = 0;


		//TODO TEST !!!
		Vec exact,diff;
		PetscViewer view;
		VecCreate(MPI_COMM_WORLD, &diff);
		VecSetSizes(diff, mesh->numPoints, PETSC_DECIDE);
		VecSetFromOptions(diff);
		
		PetscViewerBinaryOpen(PETSC_COMM_WORLD,"matlab/exact.out",FILE_MODE_READ,&view);
		VecLoad(view, VECMPI,&exact);

		do {
			j++;
			VecDuplicate(b, &br);
			VecCopy(b,br);
			for (int i = 0; i < numB; i++) {
				Vec temp;
				VecDuplicate(ghQ[i], &temp);
				VecCopy(ghQ[i], temp);
				VecAXPBY(temp, -1, r, lag[i]); 			
				MatMultTransposeAdd(B[i], temp, br, br);
				VecDestroy(temp);
			}
			

			// KROK 1 - reseni lokalnich soustav
			CGSolver cg(Ar,br,x_loc);
			cg.solve();
			cg.getX(x_loc);

			// KROK 2 - aktualizace lagrangianu
			for (int i = 0; i < numB; i++) {
				Vec temp;
				VecDuplicate(ghQ[i], &temp);
				VecCopy(ghQ[i], temp);
				VecScale(temp, -1);
				MatMultAdd(B[i], x_loc, temp, temp);
				VecAXPY(lag[i], r, temp);
			}

			//KROK 3 - update hranice

			for (int i = 0; i < numB; i++) {
				MatMult(B[i], x_loc, ghQ[i]);
				VecScale(ghQ[i], 0.5);
				VecAXPY(ghQ[i], 1,lag[i]);
				
				VecGhostRestoreLocalForm(q[indQ[i]], &ghQ[i]);
			}
			

			//
			//
			// JEDINA CAST VYZADUJICI INTERAKCI MEZI JADRY!!!!
			//
			//

			for (int i = 0; i < numQ; i++) {
				VecGhostUpdateBegin(q[i], ADD_VALUES, SCATTER_REVERSE);
				VecGhostUpdateEnd(q[i], ADD_VALUES, SCATTER_REVERSE);
				VecGhostUpdateBegin(q[i], INSERT_VALUES, SCATTER_FORWARD);
				VecGhostUpdateEnd(q[i], INSERT_VALUES, SCATTER_FORWARD);
			}

			//
			//
			// KONEC
			//
			//

			//KROK 4 - druha aktualizace lagrangianu

			PetscReal localBorderDiff = 0;
			for (int i = 0; i < numB; i++) {
				Vec temp;
				VecGhostGetLocalForm(q[indQ[i]], &ghQ[i]);
				VecDuplicate(ghQ[i], &temp);
				VecCopy(ghQ[i], temp);
				VecScale(temp, -1);
				MatMultAdd(B[i], x_loc, temp, temp);
				VecAXPY(lag[i], r, temp);

				PetscReal bordDiff;
				VecNorm(temp, NORM_MAX, &bordDiff);
				localBorderDiff += bordDiff;

	
				VecGhostGetLocalForm(borderRes[indQ[i]], &ghBorderRes[i]);
				VecCopy(lag[i], ghBorderRes[i]);
				VecGhostRestoreLocalForm(borderRes[indQ[i]], &ghBorderRes[i]);
			}

			PetscReal borderDiffNorm = 0;
			MPI_Allreduce(&localBorderDiff, &borderDiffNorm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

			VecWAXPY(diff,-1,x,exact);
			PetscReal error;
			VecNorm(diff,NORM_MAX, &error);
			
			
			//Odhad residua

			PetscReal lagNorm = 0;
			for (int i = 0; i < numQ; i++) {
				VecGhostUpdateBegin(borderRes[i], ADD_VALUES, SCATTER_REVERSE);
				VecGhostUpdateEnd(borderRes[i], ADD_VALUES, SCATTER_REVERSE);

				PetscReal tempNorm;
				VecNorm(borderRes[i], NORM_MAX, &tempNorm);
				lagNorm += tempNorm;
				
			}

			PetscPrintf(PETSC_COMM_WORLD, "%d.step\tLangrangian difference norm : %f \t Border difference norm : %f\t Error: %f\n",j, lagNorm, borderDiffNorm, error);

			VecDestroy(br);

			residualNorm = lagNorm + borderDiffNorm;
		} while (j < 50);

		MatDestroy(Ar);
		for (int i = 0; i < numB; i++)
			MatDestroy(tempMat[i]);
}

void SassiRectSystem::dumpSolution(PetscViewer v) {
	PetscObjectSetName((PetscObject)x,"u");
	VecView(x,v);
}
