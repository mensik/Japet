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

ConfigManager* ConfigManager::instance = NULL;

ConfigManager* ConfigManager::Instance() {
	if (!instance) {
		instance = new ConfigManager();
	}
	return instance;
}

ConfigManager::ConfigManager() {

	E = 2.1e5;
	mu = 0.3;
	h = 2.0;
	H = 100;
	m = 3;
	n = 3;
	problem = 0;

	PetscTruth flg;

	char tName[PETSC_MAX_PATH_LEN] = "FetiTest.log";

	PetscOptionsGetInt(PETSC_NULL, "-japet_problem", &problem, PETSC_NULL);
	PetscOptionsGetInt(PETSC_NULL, "-japet_m", &m, PETSC_NULL);
	PetscOptionsGetInt(PETSC_NULL, "-japet_n", &n, PETSC_NULL);
	PetscOptionsGetReal(PETSC_NULL, "-japet_h", &h, PETSC_NULL);
	PetscOptionsGetReal(PETSC_NULL, "-japet_HH", &H, PETSC_NULL);
	PetscOptionsGetString(PETSC_NULL, "-japet_name", tName, PETSC_MAX_PATH_LEN - 1, &flg);

	name = tName;

}

