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
#include "petscmat.h"
#include "mpi.h"
#include <time.h>
#include <sys/timeb.h>
#include <sys/resource.h>

enum CoarseProblemMethod {
	ParaCG = 0, MasterWork = 1, ORTO = 2
};

struct IterationInfo {
	int itNumber;
	PetscReal rNorm;
	std::vector<PetscReal> itData;
};

class IterationManager {
	std::string title;
	MPI_Comm comm;

	int itCounter;
	std::vector<IterationInfo> itInfo;
	std::map<std::string, PetscReal> iterationData;

	bool isVerbose;
public:
	IterationManager() {
		itCounter = 0;
		isVerbose = false;
		this->comm = MPI_COMM_SELF;
	}
	void setTitle(std::string title) {
		this->title = title;
	}
	void setComm(MPI_Comm comm) {
		this->comm = comm;
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
	ALL_ALL_SAMEROOT = 0,
	ALL_ONE_SAMEROOT = 1,
	ALL_TWO_SAMEROOT = 2,
	HECTOR = 3,
	TEST = 4,
	SAME_COMMS = 5,
	LAST_ROOT = 6
};

class PDCommManager {
	MPI_Comm parentComm;
	MPI_Comm primalComm;
	MPI_Comm dualComm;

	int parRank, pRank, dRank;
	int parSize, pSize, dSize;

public:
	PDCommManager(MPI_Comm parent, PDStrategy strategy);
	~PDCommManager() {
		if (!isSameComm()) {
			if (isPrimal()) MPI_Comm_free(&primalComm);
			if (isDual()) MPI_Comm_free(&dualComm);
		}
	}

	void printSummary();

	bool isSameComm() {

		bool iSC = false;

		if (isPrimal() && isDual()) {
			int result;
			MPI_Comm_compare(primalComm, dualComm, &result);
			iSC = result == MPI_IDENT;
		}

		return iSC;
	}

	MPI_Comm getPrimal() {
		return primalComm;
	}
	MPI_Comm getDual() {
		return dualComm;
	}
	MPI_Comm getParen() {
		return parentComm;
	}

	bool isPrimal() {
		return primalComm != MPI_COMM_NULL;
	}
	bool isDual() {
		return dualComm != MPI_COMM_NULL;
	}

	bool commonRoots() {
		return true;
	}
	bool isDualRoot() {
		return !dRank;
	}
	bool isPrimalRoot() {
		return !pRank;
	}

	int getPrimalSize() {
		return pSize;
	}
	int getDualSize() {
		return dSize;
	}

	int getPrimalRank() {
		return pRank;
	}
	int getDualRank() {
		return dRank;
	}

	void show() {
		PetscPrintf(PETSC_COMM_SELF, "[%d] ", parRank);
		if (isPrimal()) PetscPrintf(PETSC_COMM_SELF, "\t prim: %d ", pRank);
		if (isDual()) PetscPrintf(PETSC_COMM_SELF, "\t dual: %d ", dRank);
		PetscPrintf(PETSC_COMM_SELF, "\n ");
	}
};

class ConfigManager {
	static ConfigManager *instance;
	ConfigManager();
public:
	PetscInt maxIt;

	PetscInt reqSize;

	PetscInt m;
	PetscInt n;
	PetscReal h;
	PetscReal Hx;
	PetscReal Hy;

	PetscInt problem; //< 0 - Laplace, 1 - lin.elasticity

	CoarseProblemMethod coarseProblemMethod;
	PDStrategy pdStrategy;

	char *name;

	PetscReal E;
	PetscReal mu;

	bool saveOutputs;

	static ConfigManager* Instance();
};

struct MarkedTime {
	const char* title;
	PetscLogDouble time;
};

class MyTimer {
	PetscLogDouble total, start, end;
	PetscInt laps;

	std::vector<MarkedTime*> markedTimes;
public:
	MyTimer() {
		laps = 0;
		total = 0;
	}

	void startTimer() {
		PetscGetTime(&start);
	}
	void stopTimer() {
		PetscGetTime(&end);

		total += end - start;
		laps++;
	}
	void markTime(const char * title);

	PetscLogDouble getAverageTime() {
		return total / laps;
	}
	PetscInt getLapCount() {
		return laps;
	}
	PetscLogDouble getTotalTime() {
		return total;
	}

	PetscLogDouble getAverageOverComm(MPI_Comm comm);

	PetscLogDouble getMaxOverComm(MPI_Comm comm);

	void printMarkedTime(MPI_Comm comm);
};

class MyLogger {
	static MyLogger *instance;

	MyLogger() {

	}

	std::map<std::string, MyTimer*> timers;
public:
	static MyLogger* Instance();

	MyTimer* getTimer(std::string timer) {

		MyTimer* t = timers[timer];
		if (t == NULL) {
			t = new MyTimer();
			timers[timer] = t;
		}

		return t;
	}
};

//*
// Gather matrix on one (root) proces
//

void gatherMatrix(Mat A, Mat Aloc, int root, MPI_Comm comm);

#endif /* JAPETUTILS_H_ */
