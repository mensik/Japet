#include "japetUtils.h"

void IterationManager::nextIteration() {
	IterationInfo info;
	info.itNumber = itCounter;

	if (isVerbose) PetscPrintf(PETSC_COMM_WORLD, "%d: ", itCounter);

	for (std::map<std::string, PetscReal>::iterator i = iterationData.begin(); i
			!= iterationData.end(); i++) {
		const PetscReal data = i->second;
		info.itData.push_back(data);
		if (isVerbose) PetscPrintf(PETSC_COMM_WORLD, "\t%s=%f", i->first.c_str(), i->second);
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

void saveScalarResultHDF5(const char *filename, const char *name, Vec v) {
	PetscInt rank, size;
	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
	MPI_Comm_size(PETSC_COMM_WORLD, &size);

	PetscInt localSize;
	PetscReal *locArray;
	VecGetLocalSize(v, &localSize);
	VecGetArray(v, &locArray);

	if (!rank) {

		char NOM[4] = "y  ";
		char UNI[4] = "m  ";

		med_idt fid = MEDouvrir((char*) filename, MED_LECTURE_ECRITURE);
		MEDchampCr(fid, (char*) name, MED_FLOAT64, NOM, UNI, 1);

		PetscInt globalSize, procCounts[size], displ[size];

		VecGetSize(v, &globalSize);
		MPI_Gather(&localSize, 1, MPI_INT, procCounts, 1, MPI_INT, 0, PETSC_COMM_WORLD);

		displ[0] = 0;
		for (int i = 1; i < size; i++) displ[i] = displ[i - 1] + procCounts[i - 1];

		PetscReal out[globalSize];

		for (int i = 0; i < globalSize; i++) out[i] = -1;

		MPI_Gatherv(locArray, localSize, MPI_DOUBLE, out, procCounts, displ, MPI_DOUBLE, 0, PETSC_COMM_WORLD);

		MEDchampEcr(fid, "mesh", (char*) name, (unsigned char *) out, MED_FULL_INTERLACE, globalSize, MED_NOGAUSS, 1, MED_NOPFL, MED_NO_PFLMOD, MED_NOEUD, MED_NONE, MED_NOPDT, "", 0.0, MED_NONOR);
	} else {
		MPI_Gather(&localSize, 1, MPI_INT, NULL, 1, MPI_INT, 0, PETSC_COMM_WORLD);
		MPI_Gatherv(locArray, localSize, MPI_DOUBLE, NULL, NULL, NULL, MPI_DOUBLE, 0, PETSC_COMM_WORLD);
	}

	VecRestoreArray(v, &locArray);
}
