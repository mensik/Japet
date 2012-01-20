#include <iostream>
#include "mpi.h"

using namespace std;

int main(int argc, char *argv[]) {

	MPI_Init (&argc, &argv);

	{
		MPI_Comm comm = MPI_COMM_WORLD;
		int rank;
		MPI_Comm_rank(comm, &rank);

		int clusterColor = rank % 4;

		MPI_Comm subComm;
		MPI_Comm_split(comm, clusterColor, 0, &subComm);

		int subRank;
		MPI_Comm_rank(subComm, &subRank);

		MPI_Barrier(comm);
		if (rank == 0) cout << "Sync 1 \n";

		MPI_Barrier(subComm);
		if (subRank == 0) cout << "Sync Color \n";

		MPI_Barrier(comm);
		if (rank == 0) cout << "Sync 2 \n";

	}

	MPI_Finalize();
	return 0;
}
