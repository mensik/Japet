#include "fem.h"

/**
 @param[in]	R 	coordinates transformation matrix
 @param[in]  bl	local right side vector
 @param[in]	f		"force" function value
 */
void bLoc(PetscReal *R, PetscReal *bl, PetscReal f);
/**
 @param[in] R	 	coordinates transformation matrix
 @param[out] Al	local mass matrix
 @param[in] K	 	"material" function value
 */
void ALoc(PetscReal *R, PetscReal *Al, PetscReal K);

void elastLoc(Point *vetrices[], PetscReal lambda,PetscReal mu, PetscReal *fs, PetscReal *stifMat, PetscReal *bL) {

	PetscReal
			T[] =
					{ 1, 1, 1, vetrices[0]->x, vetrices[1]->x, vetrices[2]->x, vetrices[0]->y, vetrices[1]->y, vetrices[2]->y };

	PetscReal phiGrad[] = { vetrices[1]->y - vetrices[2]->y, vetrices[2]->x
			- vetrices[1]->x, vetrices[2]->y - vetrices[0]->y, vetrices[0]->x
			- vetrices[2]->x, vetrices[0]->y - vetrices[1]->y, vetrices[1]->x
			- vetrices[0]->x };

	PetscReal detT = matrixDet(T, 3);
	matrixSum(phiGrad, 1 / (2 * detT), phiGrad, 0, phiGrad, 3, 2);

	PetscReal phiGradT[6];
	matrixTranspose(phiGrad, 3, 2, phiGradT);

	double Cmu[] = { 2, 0, 0, 0, 2, 0, 0, 0, 1 };
	double Clm[] = { 1, 1, 0, 1, 1, 0, 0, 0, 0 };
	PetscReal C[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };

	matrixSum(Cmu, mu, Clm, lambda, C, 3, 3);

	double R[18], RT[18];
	for (int i = 0; i < 18; i++)
		R[i] = 0;

	int idxR1[] = { 0, 2 };
	int idxC1[] = { 0, 2, 4 };
	int idxR2[] = { 2, 1 };
	int idxC2[] = { 1, 3, 5 };

	matrixSetSub(R, 3, 6, idxR1, 2, idxC1, 3, phiGradT);
	matrixSetSub(R, 3, 6, idxR2, 2, idxC2, 3, phiGradT);

	matrixTranspose(R, 3, 6, RT);

	PetscReal temp[18];

	matrixMult(C, 3, 3, R, 3, 6, temp);
	matrixMult(RT, 6, 3, temp, 3, 6, stifMat);

	matrixSum(stifMat, detT / 2, stifMat, 0, stifMat, 6, 6);

	//Volume Force
	for (int i = 0; i < 3; i++) {
		bL[i*2] = detT / 6 * fs[0];
		bL[i*2 + 1] = detT / 6 * fs[1];
	}

	//@TODO Edge Force!!!
}

Point getCenterOfSet(Point points[], PetscInt size);

PetscErrorCode FEMAssemble2DLaplace(MPI_Comm comm, Mesh *mesh, Mat &A, Vec &b,
		PetscReal(*f)(Point), PetscReal(*K)(Point)) {
	PetscErrorCode ierr;
	PetscInt rank;
	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

	PetscInt size = mesh->vetrices.size();

	ierr
			= MatCreateMPIAIJ(comm, size, size, PETSC_DECIDE, PETSC_DECIDE, 7, PETSC_NULL, 0, PETSC_NULL, &A);
	CHKERRQ(ierr);
	ierr = VecCreateMPI(comm, size, PETSC_DECIDE, &b);
	CHKERRQ(ierr);

	std::set<PetscInt> indDirchlet;
	for (std::set<PetscInt>::iterator i = mesh->borderEdges.begin(); i
			!= mesh->borderEdges.end(); i++) {
		for (int j = 0; j < 2; j++) {
			indDirchlet.insert(mesh->edges[*i]->vetrices[j]);
		}
	}

	for (std::map<PetscInt, Element*>::iterator e = mesh->elements.begin(); e
			!= mesh->elements.end(); e++) {
		PetscScalar bl[3];
		PetscScalar Al[9];
		PetscReal R[4];

		PetscInt elSize = e->second->numVetrices;
		Point vetrices[elSize];
		PetscInt ixs[elSize];
		for (int j = 0; j < elSize; j++) {
			ixs[j] = e->second->vetrices[j];
			vetrices[j] = *(mesh->vetrices[ixs[j]]);
		}

		R[0] = vetrices[1].x - vetrices[0].x;
		R[2] = vetrices[1].y - vetrices[0].y;
		R[1] = vetrices[2].x - vetrices[0].x;
		R[3] = vetrices[2].y - vetrices[0].y;

		Point center = getCenterOfSet(vetrices, elSize);
		bLoc(R, bl, f(center));
		ALoc(R, Al, K(center));

		//Enforce Dirchlet condition
		for (int j = 0; j < 3; j++) {
			if (indDirchlet.count(ixs[j]) > 0) {
				for (int k = 0; k < 3; k++) {
					Al[j * 3 + k] = 0;
					Al[k * 3 + j] = 0;
				}
				Al[j * 3 + j] = 1;
				bl[j] = 0;
			}
		}

		ierr = VecSetValues(b, elSize, ixs, bl, ADD_VALUES);
		CHKERRQ(ierr);
		ierr = MatSetValues(A, elSize, ixs, elSize, ixs, Al, ADD_VALUES);
		CHKERRQ(ierr);
	}
	ierr = VecAssemblyBegin(b);
	CHKERRQ(ierr);
	ierr = VecAssemblyEnd(b);
	CHKERRQ(ierr);

	ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
	CHKERRQ(ierr);
	ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
	CHKERRQ(ierr);
	return ierr;

}

void FEMAssemble2DElasticity(MPI_Comm comm, Mesh *mesh, Mat &A, Vec &b) {
	PetscInt rank;
	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

	PetscInt size = mesh->vetrices.size();

	MatCreateMPIAIJ(comm, size * 2, size * 2, PETSC_DECIDE, PETSC_DECIDE, 12, PETSC_NULL, 0, PETSC_NULL, &A);
	VecCreateMPI(comm, size * 2, PETSC_DECIDE, &b);

	for (std::map<PetscInt, Element*>::iterator e = mesh->elements.begin(); e
			!= mesh->elements.end(); e++) {

		Point* vetrices[3];
		PetscInt vIndex[3];
		for (int i = 0; i < 3; i++) {
			vIndex[i] = e->second->vetrices[i];
			vetrices[i] = mesh->vetrices[vIndex[i]];
		}
		PetscReal lStiff[36];
		PetscReal bLoc[6];

		PetscReal fs[] = {0,-9.8};

		elastLoc(vetrices, 1, 3, fs, lStiff,bLoc);

		PetscInt idx[6];
		for (int i = 0; i < 3; i++) {
			idx[i * 2] = vIndex[i] * 2;
			idx[i * 2 + 1] = vIndex[i] * 2 + 1;
		}

		MatSetValues(A, 6, idx, 6, idx, lStiff, ADD_VALUES);
		VecSetValues(b, 6, idx, bLoc, ADD_VALUES);
	}

	VecAssemblyBegin(b);
	VecAssemblyEnd(b);

	MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
}

PetscErrorCode FEMAssembleTotal2DLaplace(MPI_Comm comm, Mesh *mesh, Mat &A,
		Vec &b, PetscReal(*f)(Point), PetscReal(*K)(Point)) {
	PetscErrorCode ierr;
	PetscInt rank;
	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

	PetscInt size = mesh->vetrices.size();

	ierr
			= MatCreateMPIAIJ(comm, size, size, PETSC_DECIDE, PETSC_DECIDE, 7, PETSC_NULL, 0, PETSC_NULL, &A);
	CHKERRQ(ierr);
	ierr = VecCreateMPI(comm, size, PETSC_DECIDE, &b);
	CHKERRQ(ierr);

	for (std::map<PetscInt, Element*>::iterator e = mesh->elements.begin(); e
			!= mesh->elements.end(); e++) {
		PetscScalar bl[3];
		PetscScalar Al[9];
		PetscReal R[4];

		PetscInt elSize = e->second->numVetrices;
		Point vetrices[elSize];
		PetscInt ixs[elSize];
		for (int j = 0; j < elSize; j++) {
			ixs[j] = e->second->vetrices[j];
			vetrices[j] = *(mesh->vetrices[ixs[j]]);
		}

		R[0] = vetrices[1].x - vetrices[0].x;
		R[2] = vetrices[1].y - vetrices[0].y;
		R[1] = vetrices[2].x - vetrices[0].x;
		R[3] = vetrices[2].y - vetrices[0].y;

		Point center = getCenterOfSet(vetrices, elSize);
		bLoc(R, bl, f(center));
		ALoc(R, Al, K(center));

		ierr = VecSetValues(b, elSize, ixs, bl, ADD_VALUES);
		CHKERRQ(ierr);
		ierr = MatSetValues(A, elSize, ixs, elSize, ixs, Al, ADD_VALUES);
		CHKERRQ(ierr);
	}
	ierr = VecAssemblyBegin(b);
	CHKERRQ(ierr);
	ierr = VecAssemblyEnd(b);
	CHKERRQ(ierr);

	ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
	CHKERRQ(ierr);
	ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
	CHKERRQ(ierr);
	return ierr;

}

Point getCenterOfSet(Point p[], PetscInt size) {
	PetscReal x = 0;
	PetscReal y = 0;
	PetscReal z = 0;
	for (int i = 0; i < size; i++) {
		x += p[i].x;
		y += p[i].y;
		z += p[i].z;
	}
	x /= size + 1;
	y /= size + 1;
	z /= size + 1;
	return Point(x, y, z);
}

void bLoc(PetscReal *R, PetscReal *bl, PetscReal f) {

	PetscReal dR = fabs(R[0] * R[3] - R[1] * R[2]);
	for (int i = 0; i < 3; i++)
		bl[i] = f / 6.0 * dR;
}

void ALoc(PetscReal *R, PetscReal *Al, PetscReal k) {
	PetscReal dR = fabs(R[0] * R[3] - R[1] * R[2]);
	PetscReal iR[4];
	//Transponovana inverze R
	iR[0] = -R[3] / dR;
	iR[1] = R[2] / dR;
	iR[2] = R[1] / dR;
	iR[3] = -R[0] / dR;

	PetscReal B[] =
			{ -iR[0] - iR[1], iR[0], iR[1], -iR[2] - iR[3], iR[2], iR[3] };

	//printf("%f\t%f\n",iR[0],iR[1]);
	//printf("%f\t%f\n\n",iR[2],iR[3]);

	//printf("%f\t%f\t%f\n",B[0],B[1],B[2]);
	//printf("%f\t%f\t%f\n\n",B[3],B[4],B[5]);

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			Al[i * 3 + j] = k * fabs(dR) / 2 * (B[i] * B[j] + B[i + 3] * B[j + 3]);
		}
	}
	//printf("%f\t%f\t%f\n",Al[0],Al[1],Al[2]);
	//printf("%f\t%f\t%f\n",Al[3],Al[4],Al[5]);
	//printf("%f\t%f\t%f\n",Al[6],Al[7],Al[8]);
}

