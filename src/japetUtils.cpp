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

	if (!rank) {

		hid_t file = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT);
		hid_t gCha = H5Gcreate(file, "CHA", 0);

		hid_t gResult = H5Gcreate(gCha, name, 0);

		hid_t aid, attr;
		int NCO = 1, TYP = 6;
		char NOM[] = "x";
		char UNI[] = "K";
		aid = H5Screate(H5S_SCALAR);

		hid_t strType = H5Tcopy(H5T_C_S1);
		H5Tset_size(strType, 25);

		attr = H5Acreate(gResult, "NCO", H5T_NATIVE_INT, aid, H5P_DEFAULT);
		H5Awrite(attr, H5T_NATIVE_INT, &NCO);
		H5Aclose(attr);
		attr = H5Acreate(gResult, "TYP", H5T_NATIVE_INT, aid, H5P_DEFAULT);
		H5Awrite(attr, H5T_NATIVE_INT, &TYP);
		H5Aclose(attr);
		attr = H5Acreate(gResult, "NOM", strType, aid, H5P_DEFAULT);
		H5Awrite(attr, strType, &NOM);
		H5Aclose(attr);
		attr = H5Acreate(gResult, "UNI", strType, aid, H5P_DEFAULT);
		H5Awrite(attr, strType, &UNI);
		H5Aclose(attr);

		hid_t gNOE = H5Gcreate(gResult, "NOE", 0);
		hid_t gZERO =
				H5Gcreate(gNOE, "                   0                   -1", 0);
		hid_t gMESH = H5Gcreate(gZERO, "mesh", 0);

		int size, NDT = 0, NOR = -1, NGA = 1;
		double PDT = 0.0;
		char MAI[] = "mesh", PFL[] = " ";
		VecGetSize(v, &size);

		hsize_t dim[1];
		dim[0] = (hsize_t) size;
		hid_t dataspace = H5Screate_simple(1, dim, NULL);
		hid_t datatype = H5Tcopy(H5T_IEEE_F64LE);
		hid_t dataset = H5Dcreate(gMESH, "CO", datatype, dataspace, H5P_DEFAULT);

		attr = H5Acreate(gMESH, "NBR", H5T_NATIVE_INT, aid, H5P_DEFAULT);
		H5Awrite(attr, H5T_NATIVE_INT, &size);
		H5Aclose(attr);
		attr = H5Acreate(gZERO, "NOR", H5T_NATIVE_INT, aid, H5P_DEFAULT);
		H5Awrite(attr, H5T_NATIVE_INT, &NOR);
		H5Aclose(attr);
		attr = H5Acreate(gMESH, "NGA", H5T_NATIVE_INT, aid, H5P_DEFAULT);
		H5Awrite(attr, H5T_NATIVE_INT, &NGA);
		H5Aclose(attr);
		attr = H5Acreate(gZERO, "NDT", H5T_NATIVE_INT, aid, H5P_DEFAULT);
		H5Awrite(attr, H5T_NATIVE_INT, &NDT);
		H5Aclose(attr);
		attr = H5Acreate(gZERO, "PDT", H5T_IEEE_F64LE, aid, H5P_DEFAULT);
		H5Awrite(attr, H5T_IEEE_F64LE, &PDT);
		H5Aclose(attr);
		H5Tset_size(strType, 33);
		attr = H5Acreate(gZERO, "MAI", strType, aid, H5P_DEFAULT);
		H5Awrite(attr, strType, &MAI);
		H5Aclose(attr);
		H5Tset_size(strType, 33);
		attr = H5Acreate(gMESH, "PFL", strType, aid, H5P_DEFAULT);
		H5Awrite(attr, strType, &PFL);
		H5Aclose(attr);
		attr = H5Acreate(gMESH, "GAU", strType, aid, H5P_DEFAULT);
		H5Awrite(attr, strType, &PFL);
		H5Aclose(attr);
		H5Tset_size(strType, 9);
		attr = H5Acreate(gZERO, "UNI", strType, aid, H5P_DEFAULT);
		H5Awrite(attr, strType, &UNI);
		H5Aclose(attr);

		H5Dclose(dataset);

		H5Fclose(file);
	}
}
