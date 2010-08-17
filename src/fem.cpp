#include "fem.h"

/**
	@param[in]	R 	coordinates transformation matrix
	@param[in]  bl	local right side vector
	@param[in]	f		"force" function value
*/
void bLoc(PetscScalar *R, PetscScalar *bl, PetscScalar f);
/**
	@param[in] R	 	coordinates transformation matrix
	@param[out] Al	local mass matrix
	@param[in] K	 	"material" function value
*/
void ALoc(PetscScalar *R, PetscScalar *Al, PetscScalar K);

Point getCenterOfSet(Point points[], PetscInt size);

PetscErrorCode FEMAssemble2D(MPI_Comm comm, Mesh *mesh, Mat &A, Vec &b, PetscScalar (*f)(Point), PetscScalar (*K)(Point)) {

	PetscErrorCode ierr;
	PetscMPIInt rank,numProc;
	PetscInt numOfPoints = mesh->numPoints;
	PetscInt localOwnBegin, localOwnEnd, localPointsSize;

	ierr = MatCreateMPIAIJ(comm, PETSC_DECIDE, PETSC_DECIDE, numOfPoints, numOfPoints,7,PETSC_NULL, 7, PETSC_NULL, &A);CHKERRQ(ierr);
	
	ierr = MatGetOwnershipRange(A, &localOwnBegin, &localOwnEnd); CHKERRQ(ierr);
	localPointsSize = localOwnEnd - localOwnBegin;
	ierr = VecCreate(comm, &b);CHKERRQ(ierr);
//	printf("Local size: %d\n", localPointsSize);
	ierr = VecSetSizes(b,localPointsSize, PETSC_DECIDE);CHKERRQ(ierr);
	ierr = VecSetFromOptions(b);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
	ierr = MPI_Comm_size(comm,&numProc);CHKERRQ(ierr);

	//:TODO:
	/// @todo Velmi amaterske rozdeleni sestaveni matice tuhosti mezi procesory
	///  Rad bych se pokusil casem najit reseni s vyuzitim distribuovanych
	///	poli a kazdy procesor by si sestavoval elementy podle nodu, ktere
	///	mu budou patrit => omezeni komunikace pouze na presahujici (okrajove)
	///	elementy.
		
	PetscInt elementStride = (PetscInt)floor((PetscReal)mesh->numElements / (PetscReal)numProc);
	PetscInt elStart = elementStride * rank;
	PetscInt elEnd = elStart + elementStride;
	if (elEnd > mesh->numElements) elEnd = mesh->numElements;
	
	for (PetscInt i = elStart; i < elEnd; i++) {	
		PetscScalar bl[3];
		PetscScalar Al[9];
		PetscScalar R[4];
	
		const PetscInt ixs[] = {mesh->elements[i].nodes[0],mesh->elements[i].nodes[1],mesh->elements[i].nodes[2]};
		R[0]=mesh->nodes[ixs[1]].x - mesh->nodes[ixs[0]].x ;
		R[1]=mesh->nodes[ixs[1]].y - mesh->nodes[ixs[0]].y ;
		R[2]=mesh->nodes[ixs[2]].x - mesh->nodes[ixs[0]].x ;
		R[3]=mesh->nodes[ixs[2]].y - mesh->nodes[ixs[0]].y ;
		
		bLoc(R, bl,f(mesh->elements[i].getCentroid(mesh->nodes)));
		ALoc(R, Al,K(mesh->elements[i].getCentroid(mesh->nodes)));
		
		ierr = VecSetValues(b, 3, ixs, bl, ADD_VALUES);CHKERRQ(ierr);
		ierr = MatSetValues(A, 3, ixs, 3, ixs, Al, ADD_VALUES);CHKERRQ(ierr);	
	}	
	ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
	ierr = VecAssemblyEnd(b);CHKERRQ(ierr);
	
	ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	return ierr;

}

PetscErrorCode FEMAssemble2DLaplace(MPI_Comm comm, Mesh *mesh, Mat &A, Vec &b, PetscScalar (*f)(Point), PetscScalar (*K)(Point)) {
	PetscErrorCode ierr;
	ierr = MatCreateMPIAIJ(comm, mesh->mlocal_nodes, mesh->mlocal_nodes,PETSC_DECIDE, PETSC_DECIDE, 7,PETSC_NULL, 7, PETSC_NULL, &A);CHKERRQ(ierr);
	ierr = VecCreateMPI(comm, mesh->mlocal_nodes, PETSC_DECIDE, &b);CHKERRQ(ierr);
	
	for (std::map<PetscInt, Element>::iterator i = mesh->element.begin(); i != mesh->element.end(); i++) {	
		PetscScalar bl[3];
		PetscScalar Al[9];
		PetscScalar R[4];

		PetscInt elSize = i->second.vetrices.size();
		Point vetrices[elSize];
		PetscInt ixs[elSize];
		int c = 0;
		for (std::set<PetscInt>::iterator j = i->second.vetrices.begin();
					j != i->second.vetrices.end(); j++) {
			vetrices[c] = mesh->vetrices[*j];
			ixs[c] = *j;
			c++;
		}
		
		R[0]=vetrices[1].x - vetrices[0].x ;
		R[2]=vetrices[1].y - vetrices[0].y ;
		R[1]=vetrices[2].x - vetrices[0].x ;
		R[3]=vetrices[2].y - vetrices[0].y ;
		
		Point center = getCenterOfSet(vetrices, elSize);
		bLoc(R, bl,f(center));
		ALoc(R, Al,K(center));


		
		//Enforce Dirchlet condition
		for (int j = 0; j < 3; j++) {
			if (mesh->indDirchlet.count(ixs[j]) > 0) {
				for (int k = 0; k < 3;k++) {
					Al[j*3 + k] = 0;
					Al[k*3 + j] = 0;
				}
				Al[j*3 + j] = 1;
				bl[j] = 0;
			}
		}
		
		ierr = VecSetValues(b, elSize, ixs, bl, ADD_VALUES);CHKERRQ(ierr);
		ierr = MatSetValues(A, elSize, ixs, elSize, ixs, Al, ADD_VALUES);CHKERRQ(ierr);	
	}	
	ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
	ierr = VecAssemblyEnd(b);CHKERRQ(ierr);
	
	ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	return ierr;

}

PetscErrorCode FEMAssemble2DLaplace(MPI_Comm comm, DistributedMesh *mesh, Mat &A, Vec &b, PetscScalar (*f)(Point), PetscScalar (*K)(Point)) {
	PetscErrorCode ierr;
	ierr = MatCreateMPIAIJ(comm, mesh->mlocal_nodes, mesh->mlocal_nodes,PETSC_DECIDE, PETSC_DECIDE, 7,PETSC_NULL, 7, PETSC_NULL, &A);CHKERRQ(ierr);
	ierr = VecCreateMPI(comm, mesh->mlocal_nodes, PETSC_DECIDE, &b);CHKERRQ(ierr);
	
	for (std::map<PetscInt, Element>::iterator i = mesh->element.begin(); i != mesh->element.end(); i++) {	
		PetscScalar bl[3];
		PetscScalar Al[9];
		PetscScalar R[4];

		PetscInt elSize = i->second.vetrices.size();
		Point vetrices[elSize];
		PetscInt ixs[elSize];
		int c = 0;
		for (std::set<PetscInt>::iterator j = i->second.vetrices.begin();
					j != i->second.vetrices.end(); j++) {
			vetrices[c] = mesh->vetrices[*j];
			ixs[c] = *j;
			c++;
		}
		
		R[0]=vetrices[1].x - vetrices[0].x ;
		R[2]=vetrices[1].y - vetrices[0].y ;
		R[1]=vetrices[2].x - vetrices[0].x ;
		R[3]=vetrices[2].y - vetrices[0].y ;
		
		Point center = getCenterOfSet(vetrices, elSize);
		bLoc(R, bl,f(center));
		ALoc(R, Al,K(center));


		
		//Enforce Dirchlet condition
		for (int j = 0; j < 3; j++) {
			if (mesh->indDirchlet.count(ixs[j]) > 0) {
				for (int k = 0; k < 3;k++) {
					Al[j*3 + k] = 0;
					Al[k*3 + j] = 0;
				}
				Al[j*3 + j] = 1;
				bl[j] = 0;
			}
		}
		
		ierr = VecSetValues(b, elSize, ixs, bl, ADD_VALUES);CHKERRQ(ierr);
		ierr = MatSetValues(A, elSize, ixs, elSize, ixs, Al, ADD_VALUES);CHKERRQ(ierr);	
	}	
	ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
	ierr = VecAssemblyEnd(b);CHKERRQ(ierr);
	
	ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	return ierr;

}


Point getCenterOfSet(Point p[], PetscInt size) {
	PetscScalar x = 0;
	PetscScalar y = 0;
	PetscScalar z = 0;
	for (int i = 0; i < size; i++) {
		x += p[i].x;
		y += p[i].y;
		z += p[i].z;
	}
	x /= size + 1;
	y /= size + 1;
	z /= size + 1;
	return Point(x,y,z);
}

void bLoc(PetscScalar *R, PetscScalar *bl, PetscScalar f) {
	
	PetscScalar dR = fabs(R[0]*R[3] - R[1]*R[2]);
	for (int i = 0; i < 3; i++)
		bl[i]= f / 6.0 * dR;
}

void ALoc(PetscScalar *R, PetscScalar *Al, PetscScalar k) {
	PetscScalar dR = fabs(R[0]*R[3] - R[1]*R[2]);
	PetscScalar iR[4];
	//Transponovana inverze R
	iR[0] = -R[3] / dR;
	iR[1] =  R[2] / dR;
	iR[2] =  R[1] / dR;
	iR[3] = -R[0] / dR;
	
	PetscScalar B[] = {-iR[0] - iR[1], iR[0], iR[1], -iR[2]-iR[3], iR[2], iR[3]};
	
	//printf("%f\t%f\n",iR[0],iR[1]);
	//printf("%f\t%f\n\n",iR[2],iR[3]);

	//printf("%f\t%f\t%f\n",B[0],B[1],B[2]);	
	//printf("%f\t%f\t%f\n\n",B[3],B[4],B[5]);	

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			Al[i*3 + j] = k * fabs(dR) / 2 * (B[i]*B[j] + B[i+3]*B[j+3]);
		}
	}
	//printf("%f\t%f\t%f\n",Al[0],Al[1],Al[2]);	
	//printf("%f\t%f\t%f\n",Al[3],Al[4],Al[5]);	
	//printf("%f\t%f\t%f\n",Al[6],Al[7],Al[8]);	
}

PetscErrorCode FEMSetDirchletBound(Mat &A, Vec &b, PetscInt n, PetscInt *ind) {
	PetscErrorCode ierr;

//TODO
///@todo Naji lepsi zpusob vynuceni drichletovske hranice!!!!!!!
	
	Mat temp;
	
	ierr = MatZeroRows(A,n,ind, 1);CHKERRQ(ierr);
	MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
	MatTranspose(A,MAT_INITIAL_MATRIX,&temp);
	MatDestroy(A);
	
	ierr = MatZeroRows(temp,n,ind, 1);CHKERRQ(ierr);
	MatAssemblyBegin(temp, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(temp, MAT_FINAL_ASSEMBLY);
	MatTranspose(temp,MAT_INITIAL_MATRIX,&A);

	MatDestroy(temp);
	
	//PetscInt rows,cols;
	//MatGetSize(A, &rows, &cols);
	
	for (int i = 0; i < n; i++) {
		//if (ind[i] > 4) MatSetValue(A, ind[i] - 5, ind[i], 0,INSERT_VALUES);
		//if (ind[i] > 0)MatSetValue(A, ind[i] -1 , ind[i], 0,INSERT_VALUES);
		//if (ind[i] + 1 < rows) MatSetValue(A, ind[i] + 1, ind[i], 0,INSERT_VALUES);
		//if (ind[i] + 5 < cols) MatSetValue(A, ind[i] + 5, ind[i], 0,INSERT_VALUES);
		ierr = VecSetValue(b, ind[i], 0, INSERT_VALUES);CHKERRQ(ierr);
	}

	ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
	ierr = VecAssemblyEnd(b);CHKERRQ(ierr);

	return ierr;}
