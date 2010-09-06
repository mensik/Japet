/*
 * smalbe.h
 *
 *  Created on: Aug 23, 2010
 *      Author: martin
 */

#ifndef SMALBE_H_
#define SMALBE_H_

#include <math.h>
#include <sstream>
#include <string>
#include <iostream>
#include "petscmat.h"
#include "petscmg.h"
#include "solver.h"
#include "structures.h"
#include "feti.h"

const std::string IN_ITERATION_SUFFIX = "_in.log";
const std::string OUT_ITERATION_SUFFIX = "_out.log";

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

	IterationManager itManager;

	std::string logFileName;

	PetscReal Lagrangian();

	void initPC();

public:
	Smalbe(Mat A, Vec b, Mat B, Vec c, Vec L, PetscReal mi = 1e-2, PetscReal ro = 1, PetscReal beta = 1.1, PetscReal M = 3);
	~Smalbe() ;

	void solve();
	void dumpSolution(PetscViewer v);

	void setLogFilename(std::string logFileName) { this->logFileName = logFileName; }

	void applyMult(Vec in, Vec out);
	bool isConverged(PetscInt itNum, PetscReal rNorm, Vec *vec);
	void applyPC(Vec g, Vec z);
};


#endif /* SMALBE_H_ */
