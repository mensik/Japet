#include "japetUtils.h"

void IterationManager::nextIteration() {
	IterationInfo info;
	info.itNumber = itCounter;

	if (isVerbose) PetscPrintf(comm, "%d: ", itCounter);

	for (std::map<std::string, PetscReal>::iterator i = iterationData.begin(); i
			!= iterationData.end(); i++) {
		const PetscReal data = i->second;
		info.itData.push_back(data);
		if (isVerbose) PetscPrintf(comm, "\t%s=%e", i->first.c_str(), i->second);
	}
	if (isVerbose) PetscPrintf(comm, "\n");

	itInfo.push_back(info);
	itCounter++;
}

void IterationManager::saveIterationInfo(const char *filename, bool rewrite) {
	PetscInt rank;
	MPI_Comm_rank(comm, &rank);

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
	case SAME_COMMS:
		primalComm = parentComm;
		dualComm = parentComm;
		break;
	case LAST_ROOT:
		MPI_Comm_split(parentComm, parRank > 0 ? 0 : MPI_UNDEFINED, -parRank, &primalComm);
		dualComm = primalComm;
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

void MyTimer::markTime(const char * title) {

	MarkedTime* mTime = new MarkedTime();
	mTime->title = title;

	PetscGetTime(&(mTime->time));
	mTime->time = mTime->time - start + total;
	markedTimes.push_back(mTime);
}

PetscLogDouble MyTimer::getAverageOverComm(MPI_Comm comm) {

	PetscLogDouble allTotal;
	PetscInt commSize;
	MPI_Allreduce(&total, &allTotal, 1, MPI_DOUBLE, MPI_SUM, comm);
	MPI_Comm_size(comm, &commSize);

	return allTotal / commSize;
}

PetscLogDouble MyTimer::getMaxOverComm(MPI_Comm comm) {
	PetscLogDouble max;
	MPI_Allreduce(&total, &max, 1, MPI_DOUBLE, MPI_MAX, comm);
	return max;
}

void MyTimer::printMarkedTime(MPI_Comm comm) {
	PetscPrintf(comm, "Marked times : \n");
	for (int i = 0; i < markedTimes.size(); i++) {
		PetscPrintf(comm, "%s \t\t %e s \n", markedTimes[i]->title, markedTimes[i]->time);
	}
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

	clustM = 2;
	clustN = 2;

	reqSize = 100;

	PetscTruth flg;

	coarseProblemMethod = MasterWork;
	pdStrategy = SAME_COMMS;
	saveOutputs = false;

	char tName[PETSC_MAX_PATH_LEN] = "FetiTest.log";

	PetscOptionsGetInt(PETSC_NULL, "-japet_req_size", &reqSize, PETSC_NULL);
	PetscOptionsGetInt(PETSC_NULL, "-japet_max_it", &maxIt, PETSC_NULL);
	PetscOptionsGetInt(PETSC_NULL, "-japet_problem", &problem, PETSC_NULL);
	PetscOptionsGetInt(PETSC_NULL, "-japet_m", &m, PETSC_NULL);
	PetscOptionsGetInt(PETSC_NULL, "-japet_n", &n, PETSC_NULL);
	PetscOptionsGetInt(PETSC_NULL, "-japet_clust_m", &clustM, PETSC_NULL);
		PetscOptionsGetInt(PETSC_NULL, "-japet_clust_n", &clustN, PETSC_NULL);
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

void gatherMatrix(Mat A, Mat Aloc, int root, MPI_Comm comm) {

	Mat ATemp;
	int rank;

	MPI_Comm_rank(comm, &rank);
	MatGetLocalMat(A, MAT_INITIAL_MATRIX, &ATemp);

	PetscScalar *val;
	PetscInt *ia, *ja;
	PetscInt n;
	PetscTruth done;
	PetscInt lm, ln;
	PetscInt Arows, Acols;

	MatGetSize(A, &Arows, &Acols);
	MatGetSize(ATemp, &lm, &ln);
	MatGetArray(ATemp, &val);
	MatGetRowIJ(ATemp, 0, PETSC_FALSE, PETSC_FALSE, &n, &ia, &ja, &done);

	if (rank == root) {

		PetscInt lNumRow[Arows], lNNZ[Arows], firstRowIndex[Arows + 1],
				displ[Arows], totalNNZ, totalRows;
		MPI_Gather(&n, 1, MPI_INT, lNumRow, 1, MPI_INT, 0, comm);
		MPI_Gather(&ia[n], 1, MPI_INT, lNNZ, 1, MPI_INT, 0, comm);

		totalNNZ = 0;
		totalRows = 0;

		for (int i = 0; i < Arows; i++) {
			firstRowIndex[i] = totalRows;
			displ[i] = totalNNZ;
			totalNNZ += lNNZ[i];
			totalRows += lNumRow[i];
		}
		firstRowIndex[Arows] = totalRows;

		PetscInt *locJA = new PetscInt[totalNNZ];
		PetscInt *locIA = new PetscInt[totalRows + 1];
		PetscScalar *locVal = new PetscScalar[totalNNZ];

		locIA[0] = 0;
		MPI_Gatherv(ja, ia[n], MPI_INT, locJA, lNNZ, displ, MPI_INT, 0, comm);
		MPI_Gatherv(ia + 1, n, MPI_INT, locIA + 1, lNumRow, firstRowIndex, MPI_INT, 0, comm);
		MPI_Gatherv(val, ia[n], MPI_DOUBLE, locVal, lNNZ, displ, MPI_DOUBLE, 0, comm);

		for (int j = 0; j < Arows; j++) {
			for (int i = firstRowIndex[j] + 1; i < firstRowIndex[j + 1] + 1; i++) {
				locIA[i] += displ[j];
			}
		}

		MatCreateSeqAIJWithArrays(PETSC_COMM_SELF, totalRows, ln, locIA, locJA, locVal, &Aloc);

		delete[] locJA;
		delete[] locIA;
		delete[] locVal;

	} else {
		MPI_Gather(&n, 1, MPI_INT, PETSC_NULL, 1, MPI_INT, root, comm);
		MPI_Gather(&ia[n], 1, MPI_INT, PETSC_NULL, 1, MPI_INT, root,comm);

		MPI_Gatherv(ja, ia[n], MPI_INT, NULL, NULL, NULL, MPI_INT, root, comm);
		MPI_Gatherv(ia + 1, n, MPI_INT, NULL, NULL, NULL, MPI_INT, root, comm);
		MPI_Gatherv(val, ia[n], MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE, root, comm);
	}

	MatRestoreRowIJ(ATemp, 0, PETSC_FALSE, PETSC_FALSE, &n, &ia, &ja, &done);
	MatRestoreArray(ATemp, &val);
	MatDestroy(ATemp);
}

