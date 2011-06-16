/*

 */
#ifndef MESGERR
#define MESGERR 1
#endif

#ifndef JAPETUTILS_H_
#define JAPETUTILS_H_

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include <map>
#include "petscsys.h"
#include "petscvec.h"
#include "mpi.h"
#include <time.h>
#include <sys/timeb.h>

enum CoarseProblemMethod {
	ParaCG = 0, MasterWork = 1
};

struct IterationInfo {
	int itNumber;
	PetscReal rNorm;
	std::vector<PetscReal> itData;
};

class IterationManager {
	std::string title;

	int itCounter;
	std::vector<IterationInfo> itInfo;
	std::map<std::string, PetscReal> iterationData;

	bool isVerbose;
public:
	IterationManager() {
		itCounter = 0;
		isVerbose = false;
	}
	void setTitle(std::string title) {
		this->title = title;
	}

	void nextIteration();
	void setIterationData(std::string name, PetscReal value) {
		iterationData[name] = value;
	}
	void saveIterationInfo(const char *filename, bool rewrite = true);

	void setIsVerbose(bool isVerbose) {
		this->isVerbose = isVerbose;
	}
	int getItCount() {
		return itCounter;
	}
	void reset() {
		itCounter = 0;
		itInfo.clear();
	}

};

enum PDStrategy {
	ALL_ALL_SAMEROOT = 0, ALL_ONE_SAMEROOT = 1
};

class PDCommManager {
	MPI_Comm parentComm;
	MPI_Comm primalComm;
	MPI_Comm dualComm;
public:
	PDCommManager(MPI_Comm parent, PDStrategy strategy);

	MPI_Comm getPrimal() {
		return primalComm;
	}
	MPI_Comm getDual() {
		return dualComm;
	}
	MPI_Comm getParen() {
		return parentComm;
	}
};

class ConfigManager {
	static ConfigManager *instance;
	ConfigManager();
public:
	PetscInt m;
	PetscInt n;
	PetscReal h;
	PetscReal H;

	PetscInt problem; //< 0 - Laplace, 1 - lin.elasticity

	CoarseProblemMethod coarseProblemMethod;

	char *name;

	PetscReal E;
	PetscReal mu;

	bool saveOutputs;

	static ConfigManager* Instance();
};

#endif /* JAPETUTILS_H_ */
