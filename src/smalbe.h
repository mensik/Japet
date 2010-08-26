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
#include "structures.h"

class Smalbe : public SolverApp, SolverCtr {
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
	Vec temp, tempGh;
	Vec temp2, temp2Gh;

	PetscReal Lagrangian();
public:
	Smalbe(Mat A, Vec b, Mat B, Vec c, Vec L, PetscReal mi = 1e-2, PetscReal ro = 1, PetscReal beta = 1.1, PetscReal M = 3);
	~Smalbe() ;

	void solve();
	void dumpSolution(PetscViewer v);
	void applyMult(Vec in, Vec out);
	bool isConverged(PetscInt itNum, PetscReal rNorm, Vec *vec);

};


#endif /* SMALBE_H_ */
