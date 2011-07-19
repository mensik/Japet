//============================================================================
// Name        : dayOne.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include "petscmat.h"

using namespace std;

int main(int argc, char *argv[]) {

	PetscInitialize(&argc, &argv, 0, "Day one");

	int rank, size;

	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	Vec a, b, c;

	VecCreateMPI(PETSC_COMM_WORLD, PETSC_DECIDE, 10, &a);
	VecDuplicate(a, &b);
	VecDuplicate(a, &c);

	VecSet(a, 1);

	if (!rank) {
		for (int i = 0; i < 10; i++) {
			VecSetValue(b, i, i*i, INSERT_VALUES);
		}
	}

	VecAssemblyBegin(b);
	VecAssemblyEnd(b);


	PetscViewer v;
	PetscViewerBinaryOpen(PETSC_COMM_WORLD, "b.m", FILE_MODE_WRITE, &v);
	VecView(b, v);
	PetscViewerDestroy(v);

	PetscReal normB;
	VecNorm(b, NORM_2, &normB);

	PetscPrintf(PETSC_COMM_WORLD, "b ma normu %f \n", normB);

	if (!rank) {
		PetscPrintf(PETSC_COMM_SELF, "Ale jen ja jsem master! \n");
	}

	PetscPrintf(PETSC_COMM_WORLD, "Mej se \n", rank, size);

	PetscFinalize();

	return 0;
}
