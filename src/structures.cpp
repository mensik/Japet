#include "structures.h"

PetscInt MyMultiMap::getNewPointId(PetscInt oldPointId, PetscInt domainId) {
	return data[oldPointId].count(domainId) > 0 ? data[oldPointId][domainId] : -1;
}

void MyMultiMap::saveNewPoint(PetscInt oldPointId, PetscInt domainId,
		PetscInt newPointId) {
	numOfPoints++;
	data[oldPointId].insert(std::pair<PetscInt, PetscInt>(domainId, newPointId));
}

void DomainPairings::insert(PetscInt domA, PetscInt domB, PetscInt *pair) {
	if (domA > domB) {
		PetscInt temp = domA;
		domA = domB;
		domB = temp;
	}

	data[domA][domB].push_back(pair[0]);
	data[domA][domB].push_back(pair[1]);
}

void DomainPairings::getPairs(PetscInt domA, PetscInt domB, std::vector<
		PetscInt>::iterator &begin, std::vector<PetscInt>::iterator &end) {
	if (domA > domB) {
		PetscInt temp = domA;
		domA = domB;
		domB = temp;
	}

	begin = data[domA][domB].begin();
	end = data[domA][domB].end();
}

Mesh::~Mesh() {
	PetscInt rank;
	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
	for (std::map<PetscInt, Edge*>::iterator i = edges.begin(); i != edges.end(); i++) {
		delete i->second;
	}
	for (std::map<PetscInt, Element*>::iterator i = elements.begin(); i
			!= elements.end(); i++) {
		delete i->second;
	}
	for (std::map<PetscInt, Point*>::iterator i = vetrices.begin(); i
			!= vetrices.end(); i++) {
		delete i->second;
	}

	if (!rank) {
		if (isPartitioned) {
			PetscFree(epart);
			for (std::vector<Corner*>::iterator c = corners.begin(); c
					!= corners.end(); c++) {
				delete *c;
			}
			for (std::vector<PetscInt*>::iterator p = borderPairs.begin(); p
					!= borderPairs.end(); p++) {
				delete[] *p;
			}
			delete[] startIndexes;
		}
	}
}

void Mesh::dumpForMatlab(PetscViewer v) {
	Mat x;
	Mat e;
	Vec dirch, dual, corner;

	PetscInt rank;
	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

	PetscInt indexes[] = { 0, 1, 2, 3 };
	PetscInt numVetrices = vetrices.size();

	MatCreateMPIAIJ(PETSC_COMM_WORLD, numVetrices, PETSC_DECIDE, PETSC_DECIDE, 4, 4, PETSC_NULL, 4, PETSC_NULL, &x);

	std::map<PetscInt, Point*>::iterator point = vetrices.begin();
	for (int i = 0; i < numVetrices; i++) {
		PetscInt rowNumber = point->first;

		PetscScalar data[] =
				{ point->second->x, point->second->y, point->second->z, rank };
		MatSetValues(x, 1, &rowNumber, 4, indexes, data, INSERT_VALUES);
		point++;
	}

	MatAssemblyBegin(x, MAT_FINAL_ASSEMBLY);

	PetscInt numElements = elements.size();
	MatCreateMPIAIJ(PETSC_COMM_WORLD, numElements, PETSC_DECIDE, PETSC_DECIDE, 4, 4, PETSC_NULL, 4, PETSC_NULL, &e);

	std::map<PetscInt, Element*>::iterator el = elements.begin();
	for (int i = 0; i < numElements; i++) {
		PetscInt rowNumber = el->first;
		PetscScalar data[4];
		for (int j = 0; j < el->second->numVetrices; j++)
			data[j] = el->second->vetrices[j];

		data[3] = rank;

		MatSetValues(e, 1, &rowNumber, 4, indexes, data, INSERT_VALUES);
		el++;
	}

	MatAssemblyBegin(e, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(e, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(x, MAT_FINAL_ASSEMBLY);

	//Dirchlet
	std::set<PetscInt> indDirchlet;
	for (std::set<PetscInt>::iterator i = borderEdges.begin(); i
			!= borderEdges.end(); i++) {
		for (int j = 0; j < 2; j++) {
			indDirchlet.insert(edges[*i]->vetrices[j]);
		}
	}
	VecCreateMPI(PETSC_COMM_WORLD, indDirchlet.size(), PETSC_DECIDE, &dirch);
	PetscInt localStart;
	VecGetOwnershipRange(dirch, &localStart, PETSC_NULL);
	for (std::set<PetscInt>::iterator d = indDirchlet.begin(); d
			!= indDirchlet.end(); d++) {
		VecSetValue(dirch, localStart++, *d, INSERT_VALUES);
	}
	VecAssemblyBegin(dirch);
	VecAssemblyEnd(dirch);

	if (!rank) {
		std::set<PetscInt> dualIndices;
		for (int i = 0; i < nPairs; i++) {
			dualIndices.insert(pointPairing[2 * i]);
			dualIndices.insert(pointPairing[2 * i + 1]);
		}

		VecCreateMPI(PETSC_COMM_WORLD, dualIndices.size(), PETSC_DECIDE, &dual);
		int counter = 0;
		for (std::set<PetscInt>::iterator d = dualIndices.begin(); d
				!= dualIndices.end(); d++) {
			VecSetValue(dual, counter++, *d, INSERT_VALUES);
		}

		std::set<PetscInt> cornerInd;
		for (std::vector<Corner*>::iterator corn = corners.begin(); corn
				!= corners.end(); corn++) {

			for (int i = 0; i < (*corn)->cornerSize; i++)
				cornerInd.insert((*corn)->vetrices[i]);
		}

		VecCreateMPI(PETSC_COMM_WORLD, cornerInd.size(), PETSC_DECIDE, &corner);
		counter = 0;
		for (std::set<PetscInt>::iterator d = cornerInd.begin(); d
				!= cornerInd.end(); d++) {
			VecSetValue(corner, counter++, *d, INSERT_VALUES);
		}

	} else {
		VecCreateMPI(PETSC_COMM_WORLD, 0, PETSC_DECIDE, &dual);
		VecCreateMPI(PETSC_COMM_WORLD, 0, PETSC_DECIDE, &corner);
	}
	VecAssemblyBegin(dual);
	VecAssemblyEnd(dual);
	VecAssemblyBegin(corner);
	VecAssemblyEnd(corner);
	MatView(x, v);
	MatView(e, v);
	VecView(dirch, v);
	VecView(dual, v);
	VecView(corner, v);

	MatDestroy(x);
	MatDestroy(e);
	VecDestroy(dirch);
	VecDestroy(dual);
}

void Mesh::generateRectangularMesh(PetscReal m, PetscReal n, PetscReal k,
		PetscReal l, PetscReal h) {
	PetscInt rank;
	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

	if (!rank) {
		PetscInt xEdges = (PetscInt) ceil((n - m) / h);
		PetscInt yEdges = (PetscInt) ceil((l - k) / h);

		PetscInt xPoints = xEdges + 1;
		PetscInt yPoints = yEdges + 1;

		PetscReal hx = (n - m) / xEdges;
		PetscReal hy = (l - k) / yEdges;

		//Discretization
		PetscInt nodeIndex = 0;
		PetscInt elementIndex = 0;
		for (PetscInt j = 0; j < yPoints; j++)
			for (PetscInt i = 0; i < xPoints; i++) {
				PetscReal xPos = m + i * hx;
				PetscReal yPos = k + j * hy;

				Point *node = new Point(xPos, yPos);
				vetrices.insert(std::pair<PetscInt, Point*>(nodeIndex, node));

				//Element creation
				if (j < yPoints - 1 && i < xPoints - 1) {
					Element *el1, *el2;
					el1 = new Element();
					el2 = new Element();

					el1->numVetrices = 3;
					el1->numEdges = 0;
					el1->id = elementIndex++;
					el1->vetrices[0] = nodeIndex;
					el1->vetrices[1] = nodeIndex + xPoints + 1;
					el1->vetrices[2] = nodeIndex + 1;

					el2->numVetrices = 3;
					el2->numEdges = 0;
					el2->id = elementIndex++;
					el2->vetrices[0] = nodeIndex;
					el2->vetrices[1] = nodeIndex + xPoints;
					el2->vetrices[2] = nodeIndex + xPoints + 1;

					elements.insert(std::pair<PetscInt, Element*>(el1->id, el1));
					elements.insert(std::pair<PetscInt, Element*>(el2->id, el2));
				}
				nodeIndex++;
			}
		linkPointsToElements();
		regenerateEdges();
		findBorders();

		std::set<PetscInt> dirchBorder;
		for (std::set<PetscInt>::iterator e = borderEdges.begin(); e
				!= borderEdges.end(); e++) {
			Edge *bEdge = edges[*e];
			if (vetrices[bEdge->vetrices[0]]->x > m
					|| vetrices[bEdge->vetrices[1]]->x > m) {

			} else {
				dirchBorder.insert(*e);
			}
		}

		borderEdges = dirchBorder;
	}
}

void Mesh::regenerateEdges() {

	PetscInt edgeIndex = 0;
	for (std::map<PetscInt, Element*>::iterator i = elements.begin(); i
			!= elements.end(); i++) {
		for (PetscInt j = 0; j < i->second->numVetrices; j++) {
			PetscInt v1 = i->second->vetrices[j];
			PetscInt v2 = i->second->vetrices[(j + 1) % (i->second->numVetrices)];

			PetscInt edgeInd = getEdge(v1, v2);
			if (edgeInd == -1) { // Edge doesn't exist yet
				Edge *newEdge = new Edge();
				edgeInd = edgeIndex++;
				newEdge->id = edgeInd;
				newEdge->vetrices[0] = v1;
				newEdge->vetrices[1] = v2;
				newEdge->elements.insert(i->first);
				vetrices[v1]->edges.insert(newEdge);
				vetrices[v2]->edges.insert(newEdge);
				edges.insert(std::pair<PetscInt, Edge*>(edgeInd, newEdge));
			}
			i->second->edges[i->second->numEdges++] = edgeInd;
			edges[edgeInd]->elements.insert(i->first);
		}
	}
}

void Mesh::findBorders() {
	for (std::map<PetscInt, Edge*>::iterator i = edges.begin(); i != edges.end(); i++) {
		if (i->second->elements.size() == 1) borderEdges.insert(i->first);
	}
}

PetscInt Mesh::getEdge(PetscInt nodeA, PetscInt nodeB) {
	PetscInt resultIndex = -1;
	for (std::set<Edge*>::iterator i = vetrices[nodeA]->edges.begin(); i
			!= vetrices[nodeA]->edges.end(); i++) {
		if ((*i)->vetrices[0] == nodeB || (*i)->vetrices[1] == nodeB) {
			resultIndex = (*i)->id;
			break;
		}
	}
	return resultIndex;
}

void Mesh::save(const char *filename, bool withEdges) {
	PetscInt rank;
	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

	if (!rank) {
		FILE *f;
		f = fopen(filename, "w");

		if (f != NULL) {
			fprintf(f, "Vetrices (id:x y z)\n");
			fprintf(f, "Num: %d\n", (int) vetrices.size());
			for (std::map<PetscInt, Point*>::iterator i = vetrices.begin(); i
					!= vetrices.end(); i++)
				fprintf(f, "%d:%lf %lf %lf\n", i->first, i->second->x, i->second->y, i->second->z);
			fprintf(f, "Elements (id:numVetrices [vetrices])\n");
			fprintf(f, "Num: %d\n", (int) elements.size());
			for (std::map<PetscInt, Element*>::iterator i = elements.begin(); i
					!= elements.end(); i++) {
				fprintf(f, "%d:%d", i->first, i->second->numVetrices);
				for (int j = 0; j < i->second->numVetrices; j++)
					fprintf(f, " %d", i->second->vetrices[j]);
				fprintf(f, "\n");
			}
			if (withEdges) {
				fprintf(f, "Edges (id:vetriceA vetriceB)\n");
				fprintf(f, "Num: %d\n", (int) edges.size());
				for (std::map<PetscInt, Edge*>::iterator i = edges.begin(); i
						!= edges.end(); i++)
					fprintf(f, "%d:%d %d\n", i->first, i->second->vetrices[0], i->second->vetrices[1]);
			}
			fclose(f);
		}
	}
}

void Mesh::load(const char *filename, bool withEdges) {
	PetscInt rank;
	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

	if (!rank) {
		FILE *f;
		f = fopen(filename, "r");

		if (f != NULL) {
			char msg[128];

			PetscInt numVetrices;
			fgets(msg, 128, f);
			fscanf(f, "Num: %d\n", &numVetrices);
			for (int i = 0; i < numVetrices; i++) {
				PetscInt id;
				PetscReal x, y, z;
				fscanf(f, "%d:%lf %lf %lf\n", &id, &x, &y, &z);
				Point *newPoint = new Point(x, y, z);
				vetrices.insert(std::pair<PetscInt, Point*>(id, newPoint));
			}
			PetscInt numElements;
			fgets(msg, 128, f);
			fscanf(f, "Num: %d\n", &numElements);
			for (int i = 0; i < numElements; i++) {
				PetscInt id, size;
				fscanf(f, "%d:%d", &id, &size);
				Element *newElement = new Element();
				newElement->numVetrices = size;
				for (int j = 0; j < size - 1; j++)
					fscanf(f, " %d", newElement->vetrices + j);
				fscanf(f, " %d\n", newElement->vetrices + size - 1);
				newElement->id = id;
				elements.insert(std::pair<PetscInt, Element*>(id, newElement));
			}
			if (withEdges) {
				PetscInt numEdges;
				fgets(msg, 128, f);
				fscanf(f, "Num: %d\n", &numEdges);
				for (int i = 0; i < numEdges; i++) {
					Edge *newEdge = new Edge();
					fscanf(f, "%d: %d %d\n", &(newEdge->id), newEdge->vetrices, newEdge->vetrices
							+ 1);

					vetrices[newEdge->vetrices[0]]->edges.insert(newEdge);
					vetrices[newEdge->vetrices[1]]->edges.insert(newEdge);

					edges.insert(std::pair<PetscInt, Edge*>(newEdge->id, newEdge));
				}
				for (std::map<PetscInt, Element*>::iterator i = elements.begin(); i
						!= elements.end(); i++) {
					for (PetscInt j = 0; j < i->second->numVetrices; j++) {
						PetscInt v1 = i->second->vetrices[j];
						PetscInt v2 = i->second->vetrices[(j + 1)
								% (i->second->numVetrices)];

						PetscInt edgeInd = getEdge(v1, v2);
						i->second->edges[i->second->numEdges] = edgeInd;
						edges[edgeInd]->elements.insert(i->first);
					}
				}
				findBorders();
			} else {
				regenerateEdges();
				findBorders();
			}
			linkPointsToElements();
			fclose(f);
		}
	}
}

void Mesh::linkPointsToElements() {
	for (std::map<PetscInt, Element*>::iterator i = elements.begin(); i
			!= elements.end(); i++) {
		for (int j = 0; j < i->second->numVetrices; j++) {
			vetrices[i->second->vetrices[j]]->elements.insert(i->second);
		}
	}
}

PetscErrorCode Mesh::partition(int numDomains) {

	PetscErrorCode ierr;
	PetscFunctionBegin;
	PetscInt rank, size;
	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
	MPI_Comm_size(PETSC_COMM_WORLD, &size);

	isPartitioned = true;
	numOfPartitions = numDomains;

	PetscInt NVetrices, NElements; // Global counts of vetrices and elements
	PetscInt nElements = 0; // Size of local portion of elements
	idxtype *ie, *je; // Mesh adjacency indexes

	idxtype *eDist;
	ierr = PetscMalloc((size + 1) * sizeof(idxtype), &eDist);

	if (!rank) {
		NVetrices = vetrices.size();
		NElements = elements.size();

		MPI_Bcast(&NVetrices, 1, MPI_INT, 0, PETSC_COMM_WORLD);
		MPI_Bcast(&NElements, 1, MPI_INT, 0, PETSC_COMM_WORLD);

		nElements = NElements / size;
		if (NElements % size > rank) nElements++;

	} else {
		MPI_Bcast(&NVetrices, 1, MPI_INT, 0, PETSC_COMM_WORLD);
		MPI_Bcast(&NElements, 1, MPI_INT, 0, PETSC_COMM_WORLD);

		nElements = NElements / size;
		if (NElements % size > rank) nElements++;

	}

	eDist[0] = 0;
	for (int i = 1; i < size + 1; i++) {
		idxtype elPortion = NElements / size;
		if (NElements % size > i - 1) elPortion++;
		eDist[i] = eDist[i - 1] + elPortion;
	}

	if (!rank) {

		ierr = PetscMalloc((nElements + 1) * sizeof(idxtype), &ie);
		ierr = PetscMalloc(nElements * 3 * sizeof(idxtype), &je);

		ie[0] = 0;

		std::map<PetscInt, Element*>::iterator el = elements.begin();

		for (int i = 0; i < nElements; i++, el++) {
			je[i * 3] = el->second->vetrices[0];
			je[i * 3 + 1] = el->second->vetrices[1];
			je[i * 3 + 2] = el->second->vetrices[2];

			ie[i + 1] = (i + 1) * 3;

		}

		idxtype *jE;
		ierr = PetscMalloc(nElements * 3 * sizeof(idxtype), &jE);
		for (int j = 1; j < size; j++) {
			PetscInt nE = NElements / size;
			if (NElements % size > j) nE++;
			for (int i = 0; i < nE; i++, el++) {
				jE[i * 3] = el->second->vetrices[0];
				jE[i * 3 + 1] = el->second->vetrices[1];
				jE[i * 3 + 2] = el->second->vetrices[2];
			}

			MPI_Send(jE, nE * 3, MPI_INT, j, 0, PETSC_COMM_WORLD);
		}
		ierr = PetscFree(jE);

	} else {
		ierr = PetscMalloc((nElements + 1) * sizeof(idxtype), &ie);
		ierr = PetscMalloc(nElements * 3 * sizeof(idxtype), &je);

		ie[0] = 0;
		for (int i = 1; i < nElements + 1; i++) {
			ie[i] = i * 3;
		}

		MPI_Status stat;
		MPI_Recv(je, nElements * 3, MPI_INT, 0, 0, PETSC_COMM_WORLD, &stat);
	}

	int wFlag = 0;
	int numFlag = 0;
	int nCon = 1;
	int nCommonNodes = 2;
	int options[] = { 0, 0, 0 };
	int eCut;

	float *tpwgts = new float[numDomains];
	float *ubvec = new float[numDomains];

	for (int i = 0; i < numDomains; i++) {
		tpwgts[i] = 1.0 / numDomains;
		ubvec[i] = 1.05;
	}

	idxtype *eLocPart;
	ierr = PetscMalloc(nElements * sizeof(idxtype), &eLocPart);

	ParMETIS_V3_PartMeshKway(eDist, ie, je, NULL, &wFlag, &numFlag, &nCon, &nCommonNodes, &numDomains, tpwgts, ubvec, options, &eCut, eLocPart, &PETSC_COMM_WORLD);

	ierr = PetscFree(ie);
	ierr = PetscFree(je);

	int *rCounts;

	if (!rank) {
		ierr = PetscMalloc(NElements * sizeof(idxtype), &epart);

		rCounts = new int[size];
		for (int i = 0; i < size; i++) {
			rCounts[i] = eDist[i + 1] - eDist[i];
			//	PetscPrintf(PETSC_COMM_SELF, "%d: dist: %d size: %d\n", i, eDist[i], rCounts[i]);
		}

		MPI_Gatherv(eLocPart, nElements, MPI_INT, epart, rCounts, eDist, MPI_INT, 0, PETSC_COMM_WORLD);

		delete[] rCounts;
	} else {
		MPI_Gatherv(eLocPart, nElements, MPI_INT, NULL, NULL, NULL, MPI_INT, 0, PETSC_COMM_WORLD);

	}
	ierr = PetscFree(eLocPart);
	ierr = PetscFree(eDist);
	delete[] tpwgts;
	delete[] ubvec;

	PetscFunctionReturn(ierr);
}

void Mesh::tear() {
	PetscInt rank, commSize;
	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
	MPI_Comm_size(PETSC_COMM_WORLD, &commSize);

	if (!rank) {
		MyMultiMap pointMap; // Maps original point to its image in other domain
		//  int origPointID -> { int newDomain, int newPointID }

		PetscInt edgeIDCounter = edges.size();
		PetscInt pointIDCounter = vetrices.size();

		// Algorithm goes over edges between new subdomains and performs tearing steps

		for (std::map<PetscInt, Edge*>::iterator e = edges.begin(); e
				!= edges.end(); e++) {
			if (e->second->elements.size() == 2) { // Edge is internal
				std::set<PetscInt>::iterator el = e->second->elements.begin();
				PetscInt idEl1 = *(el++);
				PetscInt idEl2 = *(el);
				if (epart[idEl1] != epart[idEl2]) { //Edge lies on the border of two subdomaines
					PetscInt idPoint[2];
					idPoint[0] = e->second->vetrices[0];
					idPoint[1] = e->second->vetrices[1];

					if (epart[idEl1] > epart[idEl2]) { //Subdomain with higher index is about to be teared
						PetscInt swap = idEl1;
						idEl1 = idEl2;
						idEl2 = swap;
					}
					PetscInt tearedSDIndex = epart[idEl2];
					PetscInt notTearedSDIndex = epart[idEl1];

					//e->second->domainInd = notTearedSDIndex;

					PetscInt idNewPoint[2];
					for (int i = 0; i < 2; i++) {
						idNewPoint[i] = pointMap.getNewPointId(idPoint[i], tearedSDIndex);
						if (idNewPoint[i] == -1) { // create NEW POINT
							Point *newPoint = new Point(*(vetrices[idPoint[i]]));
							idNewPoint[i] = pointIDCounter++;
							vetrices.insert(std::pair<PetscInt, Point*>(idNewPoint[i], newPoint));
							pointMap.saveNewPoint(idPoint[i], tearedSDIndex, idNewPoint[i]);

							//New point <-> elements
							Point *originalPoint = vetrices[idPoint[i]];
							for (std::set<Element*>::iterator pEl =
									originalPoint->elements.begin(); pEl
									!= originalPoint->elements.end(); pEl++) {
								if (epart[(*pEl)->id] == tearedSDIndex) {
									for (int iii = 0; iii < (*pEl)->numVetrices; iii++)
										if ((*pEl)->vetrices[iii] == idPoint[i]) (*pEl)->vetrices[iii]
												= idNewPoint[i];

									newPoint->elements.insert(*pEl);
									originalPoint->elements.erase(*pEl);

								}
							}
							//New point <-> edges
							for (std::set<Edge*>::iterator pEdge =
									originalPoint->edges.begin(); pEdge
									!= originalPoint->edges.end(); pEdge++) {
								std::set<PetscInt>::iterator pel = (*pEdge)->elements.begin();

								PetscInt sdIdEl1 = epart[*(pel++)];
								PetscInt sdIdEl2 = epart[*(pel)];

								if (sdIdEl1 == tearedSDIndex && ((*pEdge)->elements.size() == 1
										|| sdIdEl2 == tearedSDIndex)) { // Edge lies in new domain	(inside or on the outer border)
									for (int iii = 0; iii < 2; iii++)
										if ((*pEdge)->vetrices[iii] == idPoint[i]) (*pEdge)->vetrices[iii]
												= idNewPoint[i];

									newPoint->edges.insert(*pEdge);
									originalPoint->edges.erase(*pEdge);
								}
							}
						}
					}

					Edge *newBorderEdge = new Edge();
					newBorderEdge->id = edgeIDCounter++;
					newBorderEdge->vetrices[0] = idNewPoint[0];
					newBorderEdge->vetrices[1] = idNewPoint[1];
					newBorderEdge->domainInd = tearedSDIndex;

					vetrices[idNewPoint[0]]->edges.insert(newBorderEdge);
					vetrices[idNewPoint[1]]->edges.insert(newBorderEdge);

					newBorderEdge->elements.insert(idEl2);
					e->second->elements.erase(idEl2);

					for (int iii = 0; iii < 2; iii++) { //If any points in not taered edges was already replicated...
						PetscInt npID =
								pointMap.getNewPointId(idPoint[iii], notTearedSDIndex);
						if (npID != -1) {
							e->second->vetrices[iii] = npID;
							vetrices[npID]->edges.insert(e->second);
							vetrices[idPoint[iii]]->edges.erase(e->second);
						}
					}

					edges.insert(std::pair<PetscInt, Edge*>(newBorderEdge->id, newBorderEdge));
					PetscInt *pair = new PetscInt[2];
					pair[0] = e->first;
					pair[1] = newBorderEdge->id;
					borderPairs.push_back(pair);

					Element* element2 = elements[idEl2];
					for (int iii = 0; iii < element2->numEdges; iii++)
						if (element2->edges[iii] == e->first) element2->edges[iii]
								= newBorderEdge->id;

				}
			}
		}

		// Count vetrices,elements and edges per domain
		std::set<PetscInt> vetSetPerDom[numOfPartitions];
		PetscInt elementPerDom[numOfPartitions];
		PetscInt edgesPerDom[numOfPartitions];
		startIndexes = new PetscInt[numOfPartitions];
		for (int i = 0; i < numOfPartitions; i++)
			elementPerDom[i] = 0;
		for (int i = 0; i < numOfPartitions; i++)
			edgesPerDom[i] = 0;

		//Refresh domain numbers for vetrices and gather sets for each domain
		for (std::map<PetscInt, Element*>::iterator e = elements.begin(); e
				!= elements.end(); e++) {
			elementPerDom[epart[e->first]]++;
			for (int i = 0; i < e->second->numVetrices; i++) {
				vetSetPerDom[epart[e->first]].insert(e->second->vetrices[i]);
				vetrices[e->second->vetrices[i]]->domainInd = epart[e->first];
			}
		}

		//Refresh domain numbers for edges and gether sets of them for each domain
		for (std::map<PetscInt, Edge*>::iterator e = edges.begin(); e
				!= edges.end(); e++) {
			e->second->domainInd = vetrices[e->second->vetrices[0]]->domainInd;
			edgesPerDom[e->second->domainInd]++;
		}

		//********************************
		//			DISTRIBUTION
		//*********************************

		PetscInt indexCounter = vetSetPerDom[0].size();
		startIndexes[0] = 0;
		PetscInt elIndexCounter = elementPerDom[0];
		for (int i = 1; i < numOfPartitions; i++) {
			int nv = vetSetPerDom[i].size();
			startIndexes[i] = indexCounter;
			MPI_Send(&nv, 1, MPI_INT, i, 0, PETSC_COMM_WORLD);
			MPI_Send(&indexCounter, 1, MPI_INT, i, 0, PETSC_COMM_WORLD);
			MPI_Send(elementPerDom + i, 1, MPI_INT, i, 0, PETSC_COMM_WORLD);
			MPI_Send(&elIndexCounter, 1, MPI_INT, i, 0, PETSC_COMM_WORLD);
			MPI_Send(edgesPerDom + i, 1, MPI_INT, i, 0, PETSC_COMM_WORLD);
			indexCounter += nv;
			elIndexCounter += elementPerDom[i];
		}

		int vCounter = 0;
		PetscInt glIndices[vetSetPerDom[0].size()];
		std::map<PetscInt, Point*> masterVetrices;
		for (std::map<PetscInt, Point*>::iterator p = vetrices.begin(); p
				!= vetrices.end(); p++) {
			if (p->second->domainInd == 0) {
				masterVetrices.insert(std::pair<PetscInt, Point*>(vCounter, p->second));
				glIndices[vCounter++] = p->first;
			} else {
				int id = p->first;
				MPI_Send(&id, 1, MPI_INT, p->second->domainInd, 0, PETSC_COMM_WORLD);
				PetscReal coords[] = { p->second->x, p->second->y, p->second->z };
				MPI_Send(coords, 3, MPI_DOUBLE, p->second->domainInd, 0, PETSC_COMM_WORLD);
			}
		}

		vetrices = masterVetrices;

		AO procesOrdering;
		IS oldIS, newIS;
		ISCreateGeneral(PETSC_COMM_WORLD, vetrices.size(), glIndices, &oldIS);
		ISCreateStride(PETSC_COMM_WORLD, vetrices.size(), 0, 1, &newIS);
		AOCreateBasicIS(oldIS, newIS, &procesOrdering);

		PetscInt eCounter = 0;
		std::map<PetscInt, Element*> mastersElements;
		for (std::map<PetscInt, Element*>::iterator e = elements.begin(); e
				!= elements.end(); e++) {

			if (epart[e->first] == 0) {
				AOApplicationToPetsc(procesOrdering, e->second->numVetrices, e->second->vetrices);
				mastersElements.insert(std::pair<PetscInt, Element*>(eCounter++, e->second));
			} else {
				MPI_Send(&(e->second->numVetrices), 1, MPI_INT, epart[e->first], 0, PETSC_COMM_WORLD);

				MPI_Send(e->second->vetrices, e->second->numVetrices, MPI_INT, epart[e->first], 0, PETSC_COMM_WORLD);
			}
		}
		elements = mastersElements;

		std::map<PetscInt, Edge*> mastersEdges;
		for (std::map<PetscInt, Edge*>::iterator e = edges.begin(); e
				!= edges.end(); e++) {
			AOApplicationToPetsc(procesOrdering, 2, e->second->vetrices);
			if (e->second->domainInd == 0) {
				mastersEdges.insert(std::pair<PetscInt, Edge*>(e->first, e->second));
			} else {
				PetscInt id = e->first;
				MPI_Send(&id, 1, MPI_INT, e->second->domainInd, 0, PETSC_COMM_WORLD);
				MPI_Send(e->second->vetrices, 2, MPI_INT, e->second->domainInd, 0, PETSC_COMM_WORLD);
			}
		}

		std::set<PetscInt> mastersBorders;
		for (std::set<PetscInt>::iterator i = borderEdges.begin(); i
				!= borderEdges.end(); i++) {
			int dInd = edges[*i]->domainInd;

			if (dInd == 0) {
				mastersBorders.insert(*i);
			} else {
				PetscInt borderId = *i;

				MPI_Send(&borderId, 1, MPI_INT, dInd, 0, PETSC_COMM_WORLD);
			}
		}
		for (int i = 1; i < numOfPartitions; i++) {
			int END = -1;
			MPI_Send(&END, 1, MPI_INT, i, 0, PETSC_COMM_WORLD);
		}
		borderEdges = mastersBorders;
		edges = mastersEdges;

		nPairs = pointMap.getNumOfPoints();
		pointPairing = new PetscInt[nPairs * 2];
		int counter = 0;
		for (std::map<PetscInt, std::map<PetscInt, PetscInt> >::iterator i =
				pointMap.data.begin(); i != pointMap.data.end(); i++) {
			if (i->second.size() > 1) {
				Corner *corner = new Corner();
				corner->cornerSize = i->second.size() + 1;
				corner->vetrices[0] = i->first;
				corners.push_back(corner);
				PetscInt cCounter = 1;
				for (std::map<PetscInt, PetscInt>::iterator j = i->second.begin(); j
						!= i->second.end(); j++) {
					pointPairing[counter++] = i->first;
					pointPairing[counter++] = j->second;
					corner->vetrices[cCounter++] = j->second;
				}
				AOApplicationToPetsc(procesOrdering, corner->cornerSize, corner->vetrices);
			} else {
				for (std::map<PetscInt, PetscInt>::iterator j = i->second.begin(); j
						!= i->second.end(); j++) {
					pointPairing[counter++] = i->first;
					pointPairing[counter++] = j->second;
				}
			}
		}

		AOApplicationToPetsc(procesOrdering, nPairs * 2, pointPairing);
		MPI_Bcast(&nPairs, 1, MPI_INT, 0, PETSC_COMM_WORLD);

	} else { // Receiving of distributed mesh
		MPI_Status stats;
		PetscInt nVetrices, startIndex, elStartIndex, nElements, nEdges;
		MPI_Recv(&nVetrices, 1, MPI_INT, 0, 0, PETSC_COMM_WORLD, &stats);
		MPI_Recv(&startIndex, 1, MPI_INT, 0, 0, PETSC_COMM_WORLD, &stats);
		MPI_Recv(&nElements, 1, MPI_INT, 0, 0, PETSC_COMM_WORLD, &stats);
		MPI_Recv(&elStartIndex, 1, MPI_INT, 0, 0, PETSC_COMM_WORLD, &stats);
		MPI_Recv(&nEdges, 1, MPI_INT, 0, 0, PETSC_COMM_WORLD, &stats);
		//PetscPrintf(PETSC_COMM_SELF, "[%d] Vetrices %d , Elements %d, Edges %d n", rank, nVetrices, nElements, nEdges);
		PetscInt glIndices[nVetrices];

		for (int i = 0; i < nVetrices; i++) {
			MPI_Recv(glIndices + i, 1, MPI_INT, 0, 0, PETSC_COMM_WORLD, &stats); // RCV ID
			PetscReal coords[3];
			MPI_Recv(coords, 3, MPI_DOUBLE, 0, 0, PETSC_COMM_WORLD, &stats); // RCV COORDINATES
			vetrices.insert(std::pair<PetscInt, Point*>(i + startIndex, new Point(coords[0], coords[1], coords[2])));
		}

		AO procesOrdering;
		IS oldIS, newIS;
		ISCreateGeneral(PETSC_COMM_WORLD, nVetrices, glIndices, &oldIS);
		ISCreateStride(PETSC_COMM_WORLD, nVetrices, startIndex, 1, &newIS);
		AOCreateBasicIS(oldIS, newIS, &procesOrdering);

		for (int i = 0; i < nElements; i++) {
			Element *newEl = new Element();
			MPI_Recv(&(newEl->numVetrices), 1, MPI_INT, 0, 0, PETSC_COMM_WORLD, &stats);
			MPI_Recv(&(newEl->vetrices), newEl->numVetrices, MPI_INT, 0, 0, PETSC_COMM_WORLD, &stats);
			AOApplicationToPetsc(procesOrdering, newEl->numVetrices, newEl->vetrices);
			elements.insert(std::pair<PetscInt, Element*>(i + elStartIndex, newEl));
		}

		for (int i = 0; i < nEdges; i++) {
			PetscInt id;
			Edge *newEdge = new Edge();
			MPI_Recv(&id, 1, MPI_INT, 0, 0, PETSC_COMM_WORLD, &stats);
			MPI_Recv(newEdge->vetrices, 2, MPI_INT, 0, 0, PETSC_COMM_WORLD, &stats);
			edges.insert(std::pair<PetscInt, Edge*>(id, newEdge));
		}

		while (true) {
			int borderId;
			MPI_Recv(&borderId, 1, MPI_INT, 0, 0, PETSC_COMM_WORLD, &stats);
			if (borderId == -1) break;
			borderEdges.insert(borderId);
		}
		MPI_Bcast(&nPairs, 1, MPI_INT, 0, PETSC_COMM_WORLD);
	}

}

void Mesh::evalInNodes(PetscReal(*f)(Point), Vec *fv) {
	VecCreateMPI(PETSC_COMM_WORLD, getNumNodes(), PETSC_DECIDE, fv);

	for (std::map<PetscInt, Point*>::iterator v = vetrices.begin(); v
			!= vetrices.end(); v++) {
		VecSetValue(*fv, v->first, f(*(v->second)), INSERT_VALUES);
	}

	VecAssemblyBegin(*fv);
	VecAssemblyEnd(*fv);
}

void Mesh::analyzeDomainConection() {
	PetscInt rank;
	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
	DomainPairings pairings;
	if (!rank) {

		PetscPrintf(PETSC_COMM_WORLD, "*** ANALYZESUBDOMAINS ****\n\n");

		for (int i = 0; i < nPairs; i++) {
			PetscInt *pair = pointPairing + i * 2;
			pairings.insert(getNodeDomain(pair[0]), getNodeDomain(pair[1]), pair);
		}

		idxtype *xadj = new idxtype[numOfPartitions + 1];
		for (int i = 0; i < numOfPartitions + 1; i++)
			xadj[i] = 0;

		for (std::map<PetscInt, std::map<PetscInt, std::vector<PetscInt> > >::iterator
				i = pairings.data.begin(); i != pairings.data.end(); i++) {
			for (std::map<PetscInt, std::vector<PetscInt> >::iterator b =
					i->second.begin(); b != i->second.end(); b++) {
				//PetscPrintf(PETSC_COMM_SELF, "%d - %d : %d \n", i->first, b->first, b->second.size());
				xadj[i->first + 1]++;
				xadj[b->first + 1]++;
			}
		}
		//Sums up and finalize the xadj array
		for (int i = numOfPartitions; i >= 0; i--)
			for (int j = i + 1; j < numOfPartitions + 1; j++)
				xadj[j] += xadj[i];

		for (int i = 0; i < numOfPartitions + 1; i++) {
			PetscPrintf(PETSC_COMM_SELF, "%d \n", xadj[i]);
		}

		idxtype *adjncy = new idxtype[xadj[numOfPartitions]];
		idxtype *adjwgt = new idxtype[xadj[numOfPartitions]];

		int *tempCounter = new int[numOfPartitions];
		for (int i = 0; i < numOfPartitions; i++)
			tempCounter[i] = 0;

		for (std::map<PetscInt, std::map<PetscInt, std::vector<PetscInt> > >::iterator
				i = pairings.data.begin(); i != pairings.data.end(); i++) {
			for (std::map<PetscInt, std::vector<PetscInt> >::iterator b =
					i->second.begin(); b != i->second.end(); b++) {

				adjncy[xadj[i->first] + tempCounter[i->first]] = b->first;
				adjncy[xadj[b->first] + tempCounter[b->first]] = i->first;

				adjwgt[xadj[i->first] + tempCounter[i->first]] = b->second.size();
				adjwgt[xadj[b->first] + tempCounter[b->first]] = b->second.size();

				tempCounter[i->first]++;
				tempCounter[b->first]++;
			}
		}

		PetscPrintf(PETSC_COMM_SELF, "************** \n");
		//for (int i = 0; i < xadj[numOfPartitions]; i++) {
		//	PetscPrintf(PETSC_COMM_SELF, "%d - %d \n", adjncy[i], adjwgt[i]);
		//}

		int wgtflag = 1; //weights on edges
		int numFlag = 0; //C style
		int nparts = int(floor(sqrt(numOfPartitions)));
		int options[] = { 0, 0, 0, 0, 0 };

		int edgecut;
		idxtype *part = new idxtype[numOfPartitions];

		METIS_PartGraphKway(&numOfPartitions, xadj, adjncy, NULL, adjwgt, &wgtflag, &numFlag, &nparts, options, &edgecut, part);

		for (int i = 0; i < numOfPartitions; i++) {
			PetscPrintf(PETSC_COMM_SELF, "%d [%d] \n", i, part[i]);
		}

		delete[] tempCounter;

		delete[] xadj;
		delete[] adjncy;
		delete[] adjwgt;

		delete[] part;
	}

}

PetscInt Mesh::getNodeDomain(PetscInt index) {
	int domainIndex = 0;
	while (domainIndex + 1 < numOfPartitions) {
		if (index < startIndexes[domainIndex + 1]) break;
		domainIndex++;
	}

	return domainIndex;
}

void extractLocalAPart(Mat A, Mat *Aloc) {

	PetscInt m, n;
	MatGetOwnershipRange(A, &m, &n);
	PetscInt size = n - m;

	IS ISlocal;
	ISCreateStride(PETSC_COMM_SELF, size, m, 1, &ISlocal);
	Mat *sm;
	MatGetSubMatrices(A, 1, &ISlocal, &ISlocal, MAT_INITIAL_MATRIX, &sm);
	*Aloc = *sm;
}
