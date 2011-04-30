static char help[] = "My first own testing utility for PETSc\n\n";

#include <iostream>
#include "japetLa.h"
#include "fem.h"
#include "feti.h"

using namespace std;

int main(int argc, char *argv[]) {

	PetscInitialize(&argc, &argv, 0, help);

	Point a(0,0,0);
	Point b(1,0,0);
	Point c(0,1,0);

	Point *vetrices[] = {&a, &b, &c};

	double stifMat[36];
	PetscReal bL[6];

	PetscReal fs[] = {0, 9810 * 7.85e-9}; //V milimetrech

	elastLoc(vetrices, 2.1e5, 0.3, fs, stifMat, bL);

	display(stifMat, 6,6);

	display(bL, 6,1);

	PetscFinalize();

	return 0;
}
