#include "japetUtils.h"

void IterationManager::nextIteration() {
	IterationInfo info;
	info.itNumber = itCounter;

	if (isVerbose) PetscPrintf(PETSC_COMM_WORLD, "%d: ", itCounter);

	for (std::map<std::string, PetscReal>::iterator i = iterationData.begin(); i
			!= iterationData.end(); i++) {
		const PetscReal data = i->second;
		info.itData.push_back(data);
		if (isVerbose) PetscPrintf(PETSC_COMM_WORLD, "\t%s=%e", i->first.c_str(), i->second);
	}
	if (isVerbose) PetscPrintf(PETSC_COMM_WORLD, "\n");

	itInfo.push_back(info);
	itCounter++;
}

void IterationManager::saveIterationInfo(const char *filename, bool rewrite) {
	PetscInt rank;
	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

	if (!rank) {
		FILE *f;
		f = fopen(filename, rewrite ? "w" : "a");
		if (f != NULL) {
			fprintf(f, "# *********************************\n");
			fprintf(f, "#   %s\n", title.c_str());
			fprintf(f, "# *********************************\n");
			fprintf(f, "#\n");
			fprintf(f, "#itNumber");
			for (std::map<std::string, PetscReal>::iterator d = iterationData.begin(); d
					!= iterationData.end(); d++) {
				fprintf(f, "\t%s", d->first.c_str());
			}
			fprintf(f, "\n");

			for (std::vector<IterationInfo>::iterator i = itInfo.begin(); i
					!= itInfo.end(); i++) {

				fprintf(f, "\t%d", i->itNumber);
				for (std::vector<PetscReal>::iterator d = i->itData.begin(); d
						!= i->itData.end(); d++) {
					fprintf(f, "\t%11.4e", *d);
				}
				fprintf(f, "\n");
			}
			fprintf(f, "\n");
			fprintf(f, "\n");
			fclose(f);
		}
	}
}

PDCommManager::PDCommManager(MPI_Comm parent, PDStrategy strategy) {
	parentComm = parent;

	MPI_Comm_size(parentComm, &parSize);
	MPI_Comm_rank(parentComm, &parRank);

	switch (strategy) {
	case ALL_ALL_SAMEROOT:
		MPI_Comm_dup(parentComm, &primalComm);
		MPI_Comm_dup(parentComm, &dualComm);
		break;
	case ALL_ONE_SAMEROOT:
		MPI_Comm_dup(parentComm, &primalComm);
		MPI_Comm_split(parentComm, (parRank == 0) ? 0 : MPI_UNDEFINED, 0, &dualComm);
		break;
	case ALL_TWO_SAMEROOT:
		MPI_Comm_dup(parentComm, &primalComm);
		MPI_Comm_split(parentComm, (parRank < 2) ? 0 : MPI_UNDEFINED, 0, &dualComm);
		break;
	case HECTOR:
		MPI_Comm_dup(parentComm, &primalComm);
		MPI_Comm_split(parentComm, (parRank < 24) ? 0 : MPI_UNDEFINED, 0, &dualComm);
		break;
	case TEST:
		MPI_Comm_split(parentComm, (parRank % 3 == 0) ? 0 : MPI_UNDEFINED, 0, &primalComm);
		MPI_Comm_split(parentComm, (parRank % 2 == 0) ? 0 : MPI_UNDEFINED, 0, &dualComm);
		break;
	}

	if (isPrimal()) {
		MPI_Comm_rank(primalComm, &pRank);
		MPI_Comm_size(primalComm, &pSize);
	} else {
		pRank = -1;
		pSize = -1;
	}

	if (isDual()) {
		MPI_Comm_rank(dualComm, &dRank);
		MPI_Comm_size(dualComm, &dSize);
	} else {
		dRank = -1;
		dSize = -1;
	}

}

void PDCommManager::printSummary() {
	if (isPrimal()) PetscPrintf(PETSC_COMM_SELF, "[%d] is PRIMAL with rank %d \n", parRank, pRank);
	if (isDual()) PetscPrintf(PETSC_COMM_SELF, "[%d] is DUAL with rank %d \n", parRank, dRank);
}

ConfigManager* ConfigManager::instance = NULL;

ConfigManager* ConfigManager::Instance() {
	if (!instance) {
		instance = new ConfigManager();
	}
	return instance;
}

MyLogger* MyLogger::instance = NULL;

MyLogger* MyLogger::Instance() {
	if (!instance) {
		instance = new MyLogger();
	}
	return instance;
}

ConfigManager::ConfigManager() {

	maxIt = 60;

	E = 2.1e5;
	mu = 0.3;
	h = 2.0;
	Hx = 200;
	Hy = 100;
	m = 3;
	n = 3;
	problem = 0;

	reqSize = 100;

	PetscTruth flg;

	coarseProblemMethod = MasterWork;
	pdStrategy = HECTOR;
	saveOutputs = false;

	char tName[PETSC_MAX_PATH_LEN] = "FetiTest.log";

	PetscOptionsGetInt(PETSC_NULL, "-japet_req_size", &reqSize, PETSC_NULL);
	PetscOptionsGetInt(PETSC_NULL, "-japet_max_it", &maxIt, PETSC_NULL);
	PetscOptionsGetInt(PETSC_NULL, "-japet_problem", &problem, PETSC_NULL);
	PetscOptionsGetInt(PETSC_NULL, "-japet_m", &m, PETSC_NULL);
	PetscOptionsGetInt(PETSC_NULL, "-japet_n", &n, PETSC_NULL);
	PetscOptionsGetReal(PETSC_NULL, "-japet_h", &h, PETSC_NULL);
	PetscOptionsGetReal(PETSC_NULL, "-japet_size_x", &Hx, PETSC_NULL);
	PetscOptionsGetReal(PETSC_NULL, "-japet_size_y", &Hy, PETSC_NULL);
	PetscOptionsGetString(PETSC_NULL, "-japet_name", tName, PETSC_MAX_PATH_LEN
			- 1, &flg);
	PetscOptionsGetInt(PETSC_NULL, "-japet_cpmethod", (PetscInt*) &coarseProblemMethod, PETSC_NULL);
	PetscOptionsGetInt(PETSC_NULL, "-japet_pd_strategy", (PetscInt*) &pdStrategy, PETSC_NULL);
	PetscOptionsGetTruth(PETSC_NULL, "-japet_save_output", (PetscTruth*) &saveOutputs, PETSC_NULL);

	PetscOptionsGetReal(PETSC_NULL, "-japet_e", &E, PETSC_NULL);
	PetscOptionsGetReal(PETSC_NULL, "-japet_mu", &mu, PETSC_NULL);

	name = tName;

}

