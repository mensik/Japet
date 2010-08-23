/*
 * smalbe.h
 *
 *  Created on: Aug 23, 2010
 *      Author: martin
 */

#ifndef SMALBE_H_
#define SMALBE_H_

#include <math.h>
#include "petscmat.h"
#include "solver.h"

class Smalbe : public SolverApp {
	Mat A;
	Mat B;
	Vec b;
	Vec x;
	Vec lmb;
	Vec c;
	Vec L;

	PetscReal mi;
	PetscReal ro;
	PetscReal beta;
	PetscReal M;

	//TEMPS
	Vec tempMSize;
	Vec temp;

public:
	Smalbe(Mat A, Vec b, Mat B, Vec c, Vec L, PetscReal mi = 1e-3, PetscReal ro = 5, PetscReal beta = 1.1, PetscReal M = 0.1);
	~Smalbe() ;

	void solve();
	void dumpSolution(PetscViewer v);
	void applyMult(Vec in, Vec out);

};


#endif /* SMALBE_H_ */
