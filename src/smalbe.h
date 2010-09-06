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
#include "petscmg.h"
#include "solver.h"
#include "structures.h"
#include "feti.h"

class Smalbe : public SolverApp, SolverCtr, SolverPreconditioner {
	Mat A;
	Mat B;
	Mat Bloc;
	Vec b;
	Vec x;
	Vec lmb;
	Vec c;
	Vec L;

	Mat AroLoc;
	PC pc;

	PetscReal mi;
	PetscReal ro;
	PetscReal beta;
	PetscReal M;

	//TEMPS
	Vec tempMSize;
	Vec temp, tempGh;
	Vec temp2, temp2Gh;

	PetscReal Lagrangian();

	void initPC();

public:
	Smalbe(Mat A, Vec b, Mat B, Vec c, Vec L, PetscReal mi = 1e-2, PetscReal ro = 1, PetscReal beta = 1.1, PetscReal M = 3);
	~Smalbe() ;

	void solve();
	void dumpSolution(PetscViewer v);

	void applyMult(Vec in, Vec out);
	bool isConverged(PetscInt itNum, PetscReal rNorm, Vec *vec);
	void applyPC(Vec g, Vec z);
};


#endif /* SMALBE_H_ */
