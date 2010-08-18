#include "structures.h"


PetscInt MyMultiMap::getNewPointId(PetscInt oldPointId,PetscInt domainId) {
	return data[oldPointId].count(domainId) > 0?data[oldPointId][domainId]:-1;	
}

void MyMultiMap::saveNewPoint(PetscInt oldPointId, PetscInt domainId, PetscInt newPointId) {
	numOfPoints++;
	data[oldPointId].insert(std::pair<PetscInt, PetscInt>(domainId, newPointId));
}
 
Mesh::~Mesh() {
	for (std::map<PetscInt,Edge*>::iterator i = edges.begin(); i != edges.end(); i++) {
		delete i->second;
	}
	for (std::map<PetscInt,Element*>::iterator i = elements.begin(); i != elements.end(); i++) {
		delete i->second;
	}
	for (std::map<PetscInt,Point*>::iterator i = vetrices.begin(); i != vetrices.end(); i++) {
		delete i->second;
	}
	if (isPartitioned) delete [] epart; 
}

void DistributedMesh::dumpForMatlab(PetscViewer v) {
	Mat x;
	Mat e;

	PetscInt rank;
	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

	PetscInt indexes[] = {0,1,2,3};

	MatCreateMPIAIJ(PETSC_COMM_WORLD, nVetrices, PETSC_DECIDE, PETSC_DECIDE, 4, 4,PETSC_NULL, 4,PETSC_NULL, &x);
	for (int i = 0; i < nVetrices;i++) {
		PetscInt rowNumber = i + startIndex;
		PetscScalar	data[] = {vetrices[i].x, vetrices[i].y, vetrices[i].z, rank};
		MatSetValues(x,1, &rowNumber,4, indexes ,data,INSERT_VALUES);
	}
	MatAssemblyBegin(x,MAT_FINAL_ASSEMBLY);
	MatCreateMPIAIJ(PETSC_COMM_WORLD, nElements, PETSC_DECIDE, PETSC_DECIDE, 4,4,PETSC_NULL, 4, PETSC_NULL, &e);	
	
	for (int i = 0; i < nElements;i++) {
		PetscScalar data[4];
		for (int j = 0; j < elements[i].numVetrices; j++) {
			data[j] = elements[i].vetrices[j];
		}
		data[3] = rank;	
		//PetscPrintf(PETSC_COMM_SELF, "[%f] %f %f %f\n", data[3], data[0], data[1], data[2]);
		PetscInt rowNumber = i + elStartIndex;
		MatSetValues(e, 1,&rowNumber,4, indexes ,data,INSERT_VALUES);
	}
	MatAssemblyBegin(e,MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(e, MAT_FINAL_ASSEMBLY);
	MatAssemblyEnd(x, MAT_FINAL_ASSEMBLY);
	
	MatView(x,v);
	MatView(e,v);	

	MatDestroy(x);
	MatDestroy(e);
}

void Mesh::dumpForMatlab(PetscViewer v) {
	Mat x;
	Mat e;

	PetscInt rank;
	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

	if (!rank) {
		PetscInt indexes[] = {0,1,2,3};
		PetscInt numVetrices = vetrices.size();

		MatCreateMPIAIJ(PETSC_COMM_WORLD, numVetrices, PETSC_DECIDE, PETSC_DECIDE, 4, 4,PETSC_NULL, 4,PETSC_NULL, &x);
		
		std::map<PetscInt, Point*>::iterator point = vetrices.begin();
		for (int i = 0; i < numVetrices;i++) {
			PetscInt rowNumber = point->first;
			
			
			PetscScalar	data[] = {point->second->x, point->second->y, point->second->z, isPartitioned?point->second->domainInd:0};
			MatSetValues(x,1, &rowNumber,4, indexes ,data,INSERT_VALUES);
			point++;
		}
		MatAssemblyBegin(x,MAT_FINAL_ASSEMBLY);
		PetscInt numElements = elements.size();
		MatCreateMPIAIJ(PETSC_COMM_WORLD, numElements, PETSC_DECIDE, PETSC_DECIDE, 4,4,PETSC_NULL, 4, PETSC_NULL, &e);	
		
		std::map<PetscInt, Element*>::iterator el = elements.begin();
		for (int i = 0; i < numElements;i++) {
			PetscInt rowNumber = el->first;
			PetscScalar data[4];
			PetscInt counter = 0;
			for (int j = 0; j < el->second->numVetrices; j++)
				data[counter++] = el->second->vetrices[j];
			
			data[3] = 0;
			if (isPartitioned) data[3] = epart[el->first];
			MatSetValues(e,1, &rowNumber,4, indexes ,data,INSERT_VALUES);
			el++;
		}
		MatAssemblyBegin(e,MAT_FINAL_ASSEMBLY);
		MatAssemblyEnd(e, MAT_FINAL_ASSEMBLY);
		MatAssemblyEnd(x, MAT_FINAL_ASSEMBLY);
	} else {
		MatCreateMPIAIJ(PETSC_COMM_WORLD, 0, PETSC_DECIDE, PETSC_DECIDE, 4, 4,PETSC_NULL, 4,PETSC_NULL, &x);
		MatAssemblyBegin(x,MAT_FINAL_ASSEMBLY);
		MatCreateMPIAIJ(PETSC_COMM_WORLD, 0, PETSC_DECIDE, PETSC_DECIDE, 4,4,PETSC_NULL, 4, PETSC_NULL, &e);	
		MatAssemblyBegin(e,MAT_FINAL_ASSEMBLY);
		MatAssemblyEnd(e, MAT_FINAL_ASSEMBLY);
		MatAssemblyEnd(x, MAT_FINAL_ASSEMBLY);
	}
	MatView(x,v);
	MatView(e,v);	

	MatDestroy(x);
	MatDestroy(e);
}

void Mesh::generateRectangularMesh(PetscReal m, PetscReal n, PetscReal k, PetscReal l, PetscReal h) {
	PetscInt rank;
	MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

	if (!rank) {
	PetscInt xEdges = (PetscInt)ceil((n - m) / h);
	PetscInt yEdges = (PetscInt)ceil((l - k) / h);

	PetscInt xPoints = xEdges + 1;
	PetscInt yPoints = yEdges + 1;

	PetscReal hx = (n - m) / xEdges;
	PetscReal hy = (l - k) / yEdges;

	//Discretization
	PetscInt nodeIndex = 0;
	PetscInt elementIndex = 0;
	for (PetscInt j = 0; j < yPoints; j++)
		for (PetscInt i = 0; i < xPoints; i++) {
				PetscReal xPos = m + i*hx;
				PetscReal yPos = k + j*hy;			

				Point *node = new Point(xPos,yPos);
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
					el2->vetrices[2] = nodeIndex + xPoints +	 1;

					elements.insert(std::pair<PetscInt, Element*>(el1->id, el1));
					elements.insert(std::pair<PetscInt, Element*>(el2->id, el2));
				}
				nodeIndex++;
		}
		linkPointsToElements();
		regenerateEdges();
		findBorders();
	}
}

void Mesh::regenerateEdges() {

	PetscInt edgeIndex = 0;
	for (std::map<PetscInt, Element*>::iterator i = elements.begin(); i != elements.end(); i++) {
		for (PetscInt j = 0; j < i->second->numVetrices; j++) {
			PetscInt v1 = i->second->vetrices[j];
			PetscInt v2 = i->second->vetrices[(j+1) % (i->second->numVetrices)];
			
			PetscInt edgeInd = getEdge(v1,v2);
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
	for (std::set<Edge*>::iterator i = vetrices[nodeA]->edges.begin(); i != vetrices[nodeA]->edges.end(); i++) {
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
			fprintf(f, "Num: %d\n", (int)vetrices.size());
			for (std::map<PetscInt, Point*>::iterator i = vetrices.begin(); i != vetrices.end(); i++)
				fprintf(f, "%d:%lf %lf %lf\n", i->first, i->second->x, i->second->y, i->second->z);
			fprintf(f, "Elements (id:numVetrices [vetrices])\n");
			fprintf(f, "Num: %d\n", (int)elements.size());
			for (std::map<PetscInt, Element*>::iterator i = elements.begin(); i != elements.end(); i++) {
				fprintf(f, "%d:%d", i->first, i->second->numVetrices);
				for (int j = 0; j < i->second->numVetrices; j++)
					fprintf(f, " %d", i->second->vetrices[j]);
				fprintf(f, "\n");
			}
			if (withEdges) {
				fprintf(f, "Edges (id:vetriceA vetriceB)\n");
				fprintf(f, "Num: %d\n", (int)edges.size());
				for (std::map<PetscInt, Edge*>::iterator i = edges.begin(); i != edges.end(); i++)
					fprintf(f,"%d:%d %d\n", i->first, i->second->vetrices[0], i->second->vetrices[1]);
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
				PetscReal x,y,z;
				fscanf(f, "%d:%lf %lf %lf\n",&id, &x, &y,&z);
				Point *newPoint = new Point(x,y,z);
				vetrices.insert(std::pair<PetscInt, Point*>(id, newPoint));
			}
			PetscInt numElements;
			fgets(msg, 128, f);
			fscanf(f, "Num: %d\n", &numElements);
			for (int i = 0; i < numElements; i++) {
				PetscInt id, size;
				fscanf(f, "%d:%d",&id, &size);
				Element *newElement = new Element();
				newElement->numVetrices = size;
				for (int j = 0; j < size-1; j++)
					fscanf(f," %d",newElement->vetrices + j);
				fscanf(f," %d\n", newElement->vetrices + size-1);
				newElement->id = id;
				elements.insert(std::pair<PetscInt, Element*>(id, newElement));	
			}
			if (withEdges) {
				PetscInt numEdges;
				fgets(msg, 128, f);
				fscanf(f, "Num: %d\n", &numEdges);
				for (int i = 0; i < numEdges; i++) {
					Edge *newEdge = new Edge();
					fscanf(f, "%d: %d %d\n", &(newEdge->id), newEdge->vetrices, newEdge->vetrices + 1);
					
					vetrices[newEdge->vetrices[0]]->edges.insert(newEdge);
					vetrices[newEdge->vetrices[1]]->edges.insert(newEdge);
					
					edges.insert(std::pair<PetscInt, Edge*>(newEdge->id, newEdge));
				}
				for (std::map<PetscInt, Element*>::iterator i = elements.begin(); i != elements.end(); i++) {
					for (PetscInt j = 0; j < i->second->numVetrices; j++) {
						PetscInt v1 = i->second->vetrices[j];
						PetscInt v2 = i->second->vetrices[(j+1) % (i->second->numVetrices)];
			
						PetscInt edgeInd = getEdge(v1,v2);
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
	for (std::map<PetscInt, Element*>::iterator i = elements.begin(); i != elements.end(); i++) {
		for (int j = 0; j < i->second->numVetrices; j++) {
			vetrices[i->second->vetrices[j]]->elements.insert(i->second);
		}
	}
}

void Mesh::partition(int numDomains) {
	//TODO now only for triangles

	PetscInt rank;
	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

	if (!rank) {	
	isPartitioned = true;
	numOfPartitions = numDomains;

	int eType = 1;
	int numFlag = 0;
	int nn = vetrices.size();
	int ne = elements.size();
	int edgeCut;
	idxtype npart[nn];
	idxtype elmnts[ne * 3];
	
	epart = new idxtype[ne];
	int i = 0;
	for (std::map<PetscInt, Element*>::iterator el = elements.begin(); el != elements.end(); el++) {
		elmnts[i++] = el->second->vetrices[0];
		elmnts[i++] = el->second->vetrices[1];
		elmnts[i++] = el->second->vetrices[2];
	}

	METIS_PartMeshDual(&ne, &nn, elmnts, &eType, &numFlag, &numDomains, &edgeCut, epart, npart);
	
	//for (std::map<PetscInt, Point*>::iterator v = vetrices.begin(); v != vetrices.end(); v++) {
	//	v->second->domainInd = npart[v->first];
	//}
	}
}

void Mesh::tear(DistributedMesh *dm) {
	PetscInt rank, commSize;
	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
	MPI_Comm_size(PETSC_COMM_WORLD, &commSize);

	if (!rank) {
	MyMultiMap pointMap; // Maps original point to its image in other domain
																									//  int origPointID -> { int newDomain, int newPointID }
	std::map<PetscInt, PetscInt> borderPairings; //Pair borders together

	PetscInt edgeIDCounter = edges.size();
	PetscInt pointIDCounter = vetrices.size();
	
		// Algorithm goes over edges between new subdomains and performs tearing steps
		
		for (std::map<PetscInt, Edge*>::iterator e = edges.begin(); e != edges.end(); e++) {
			if (e->second->elements.size() == 2) { // Edge is internal
				std::set<PetscInt>::iterator el = e->second->elements.begin();
				PetscInt idEl1 = *(el++);
				PetscInt idEl2 = *(el);
				if (epart[idEl1] != epart[idEl2]) { //Edge lies on the border of two subdomaines
					PetscInt idPoint[2];
					idPoint[0] = e->second->vetrices[0];
					idPoint[1] = e->second->vetrices[1];

					if (epart[idEl1]>epart[idEl2]) { //Subdomain with higher index is about to be teared
						PetscInt swap = idEl1;
						idEl1 = idEl2;
						idEl2 = swap;					
					}
					PetscInt tearedSDIndex = epart[idEl2];
					PetscInt notTearedSDIndex = epart[idEl1];
				
					//e->second->domainInd = notTearedSDIndex;
					
					PetscInt idNewPoint[2];
					for (int i = 0; i < 2; i++) {
						idNewPoint[i] = pointMap.getNewPointId(idPoint[i],tearedSDIndex);
						if (idNewPoint[i] == -1) { // create NEW POINT
							Point *newPoint = new Point(*(vetrices[idPoint[i]]));
							idNewPoint[i] = pointIDCounter++;
							vetrices.insert(std::pair<PetscInt, Point*>(idNewPoint[i], newPoint));
							pointMap.saveNewPoint(idPoint[i], tearedSDIndex, idNewPoint[i]);
							
							//New point <-> elements
							Point *originalPoint = vetrices[idPoint[i]];
							for (std::set<Element*>::iterator pEl = originalPoint->elements.begin(); pEl != originalPoint->elements.end(); pEl++) {
								if (epart[(*pEl)->id] == tearedSDIndex) {
									for (int iii = 0; iii < (*pEl)->numVetrices; iii++)
										if ((*pEl)->vetrices[iii] == idPoint[i])
											(*pEl)->vetrices[iii] = idNewPoint[i];

									newPoint->elements.insert(*pEl);
									originalPoint->elements.erase(*pEl);

								}
							}
							//New point <-> edges
							for (std::set<Edge*>::iterator pEdge = originalPoint->edges.begin(); pEdge != originalPoint->edges.end(); pEdge++) {
										std::set<PetscInt>::iterator pel = (*pEdge)->elements.begin();
										
										PetscInt sdIdEl1 = epart[*(pel++)];
										PetscInt sdIdEl2 = epart[*(pel)];

										if (sdIdEl1 == tearedSDIndex && ((*pEdge)->elements.size() == 1 || sdIdEl2 == tearedSDIndex)) { // Edge lies in new domain	(inside or on the outer border)	
											for (int iii = 0; iii < 2; iii++)
												if ((*pEdge)->vetrices[iii] == idPoint[i]) (*pEdge)->vetrices[iii] = idNewPoint[i];
								
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
						PetscInt npID = pointMap.getNewPointId(idPoint[iii], notTearedSDIndex);
						if (npID != -1) {
							e->second->vetrices[iii] = npID;
							vetrices[npID]->edges.insert(e->second);
							vetrices[idPoint[iii]]->edges.erase(e->second);
						}
					}

					edges.insert(std::pair<PetscInt, Edge*>(newBorderEdge->id, newBorderEdge));
					borderPairings.insert(std::pair<PetscInt,PetscInt>(e->first, newBorderEdge->id));			
					
					Element* element2 = elements[idEl2];
					for (int iii = 0; iii < element2->numEdges; iii++)
						if (element2->edges[iii] == e->first) element2->edges[iii] = newBorderEdge->id;



				}
			}
		}
		

		// Count vetrices a edges per domain
		std::set<PetscInt> vetSetPerDom[numOfPartitions];
		PetscInt elementPerDom[numOfPartitions];
		for (int i = 0; i < numOfPartitions; i++) elementPerDom[i] = 0;

		for (std::map<PetscInt, Element*>::iterator e = elements.begin(); e != elements.end(); e++) {
			elementPerDom[epart[e->first]]++;
			for (int i = 0; i < e->second->numVetrices; i++) {
				vetSetPerDom[epart[e->first]].insert(e->second->vetrices[i]);
				vetrices[e->second->vetrices[i]]->domainInd = epart[e->first];
			}
		}

		//Distribution to master
		dm->nVetrices = vetSetPerDom[0].size();
		dm->nElements = elementPerDom[0];
		dm->vetrices = new Point[dm->nVetrices];
		dm->elements = new Element[dm->nElements];
		PetscInt glIndices[dm->nVetrices];
		dm->startIndex = 0;
		dm->elStartIndex = 0;
		
		PetscInt indexCounter = vetSetPerDom[0].size();
		PetscInt elIndexCounter = elementPerDom[0];
		for (int i = 1; i < numOfPartitions; i++) {
			int nv = vetSetPerDom[i].size();
			MPI_Send(&nv, 1, MPI_INT, i, 0, PETSC_COMM_WORLD);
			MPI_Send(elementPerDom + i, 1, MPI_INT, i, 0, PETSC_COMM_WORLD);
			MPI_Send(&indexCounter, 1, MPI_INT, i, 0, PETSC_COMM_WORLD);
			MPI_Send(&elIndexCounter, 1, MPI_INT, i, 0, PETSC_COMM_WORLD);
			indexCounter += nv;
			elIndexCounter += elementPerDom[i];
		}
  	int vCounter = 0;
		for (std::map<PetscInt, Point*>::iterator p = vetrices.begin(); p != vetrices.end(); p++) {
			if (p->second->domainInd == 0) {
				dm->vetrices[vCounter] = Point(p->second->x, p->second->y, p->second->z);
				glIndices[vCounter] = p->first;
				vCounter++;						
			} else {
				int id = p->first;
				MPI_Send(&id, 1, MPI_INT, p->second->domainInd, 0, PETSC_COMM_WORLD); 
				PetscScalar coords[] = {p->second->x, p->second->y, p->second->z};
				MPI_Send(coords, 3, MPI_DOUBLE, p->second->domainInd, 0, PETSC_COMM_WORLD);
			}
		}
		
		IS oldIS, newIS;
		ISCreateGeneral(PETSC_COMM_WORLD, dm->nVetrices, glIndices, &oldIS);
		ISCreateStride(PETSC_COMM_WORLD, dm->nVetrices, dm->startIndex,1,&newIS);
		AOCreateBasicIS(oldIS, newIS, &(dm->procesOrdering));

		int eCounter = 0;
		for (std::map<PetscInt, Element*>::iterator e = elements.begin(); e != elements.end(); e++) {
			if (epart[e->first] == 0) {
				dm->elements[eCounter].numVetrices = e->second->numVetrices;
				for (int i = 0; i < dm->elements[eCounter].numVetrices; i++) dm->elements[eCounter].vetrices[i] = e->second->vetrices[i];
				AOApplicationToPetsc(dm->procesOrdering, dm->elements[eCounter].numVetrices, dm->elements[eCounter].vetrices);
				eCounter++;
			} else {
				MPI_Send(&(e->second->numVetrices), 1, MPI_INT, epart[e->first], 0, PETSC_COMM_WORLD);
				MPI_Send(e->second->vetrices, e->second->numVetrices, MPI_INT, epart[e->first], 0, PETSC_COMM_WORLD);	
			}
		}
		for (std::set<PetscInt>::iterator i = borderEdges.begin(); i != borderEdges.end(); i++) {
			for (int j = 0; j < 2; j++) {
				int idBorder = edges[*i]->vetrices[j];
				int dInd = vetrices[idBorder]->domainInd;
				if (dInd == 0) {
					AOApplicationToPetsc(dm->procesOrdering, 1, &idBorder);
					dm->indDirchlet.insert(idBorder);
				} else {
					MPI_Send(&idBorder, 1, MPI_INT, dInd, 0, PETSC_COMM_WORLD);
				}
			}
		}
		for (int i = 1; i < numOfPartitions; i++) {
			int END = -1;
			MPI_Send(&END, 1, MPI_INT, i, 0, PETSC_COMM_WORLD);

		}

		dm->nPairs = pointMap.getNumOfPoints();
		dm->pointPairing = new PetscInt[dm->nPairs * 2];
		int counter = 0;
		for (std::map<PetscInt, std::map<PetscInt,PetscInt> >::iterator i = pointMap.data.begin(); i != pointMap.data.end(); i++) {
			for (std::map<PetscInt,PetscInt>::iterator j = i->second.begin(); j != i->second.end(); j++) {
					dm->pointPairing[counter++] = i->first;
					dm->pointPairing[counter++] = j->second;
			}
		}
		AOApplicationToPetsc(dm->procesOrdering, dm->nPairs * 2, dm->pointPairing);
		MPI_Bcast(&(dm->nPairs), 1, MPI_INT, 0, PETSC_COMM_WORLD);

		/*  Print 

		for (std::map<PetscInt, Point*>::iterator p = vetrices.begin(); p != vetrices.end(); p++) {
			PetscPrintf(PETSC_COMM_SELF, "Point %d (%d) - edges: ", p->first, p->second->domainInd);
			for (std::set<Edge*>::iterator e = p->second->edges.begin(); e != p->second->edges.end(); e++) {
				PetscPrintf(PETSC_COMM_SELF, "%d ", (*e)->id);
			}
			PetscPrintf(PETSC_COMM_SELF, "  \t elements: ");
			for (std::set<Element*>::iterator e = p->second->elements.begin(); e != p->second->elements.end();e++) {
				PetscPrintf(PETSC_COMM_SELF, "%d ", (*e)->id);
			}
			PetscPrintf(PETSC_COMM_SELF, "\n");
		}

		for (std::map<PetscInt, Edge*>::iterator e = edges.begin(); e != edges.end(); e++) {
			PetscPrintf(PETSC_COMM_SELF, "Edge %d (%d-%d) ", e->first, e->second->vetrices[0], e->second->vetrices[1]);
			PetscPrintf(PETSC_COMM_SELF, " \t elements: ");
			for (std::set<PetscInt>::iterator el = e->second->elements.begin(); el != e->second->elements.end(); el++) {
				PetscPrintf(PETSC_COMM_SELF, "%d ", *el);
			}
			PetscPrintf(PETSC_COMM_SELF, "\n");
		}

		for (std::map<PetscInt, Element*>::iterator e = elements.begin(); e != elements.end(); e++) {
			PetscPrintf(PETSC_COMM_SELF, "Element %d - (%d, %d, %d) edges %d %d %d\n", e->first, e->second->vetrices[0],  e->second->vetrices[1],  e->second->vetrices[2],  e->second->edges[0],  e->second->edges[1],  e->second->edges[2] );
		}
	*/

	} else { // Receiving of distributed mesh
		MPI_Status stats;
		MPI_Recv(&(dm->nVetrices), 1, MPI_INT, 0, 0, PETSC_COMM_WORLD, &stats);
		MPI_Recv(&(dm->nElements), 1, MPI_INT, 0, 0, PETSC_COMM_WORLD, &stats);
		MPI_Recv(&(dm->startIndex), 1, MPI_INT, 0,0, PETSC_COMM_WORLD, &stats);
		MPI_Recv(&(dm->elStartIndex), 1, MPI_INT, 0, 0, PETSC_COMM_WORLD, &stats);

		dm->vetrices = new Point[dm->nVetrices];
		dm->elements = new Element[dm->nElements];
		PetscInt glIndices[dm->nVetrices];

		for (int i = 0; i < dm->nVetrices; i++) {
			MPI_Recv(glIndices + i, 1, MPI_INT, 0, 0, PETSC_COMM_WORLD, &stats); // RCV ID
			PetscScalar coords[3];
			MPI_Recv(coords, 3, MPI_DOUBLE, 0, 0, PETSC_COMM_WORLD, &stats); // RCV COORDINATES
			dm->vetrices[i].x = coords[0];
			dm->vetrices[i].y = coords[1];
			dm->vetrices[i].z = coords[2];
		}
		IS oldIS, newIS;
		ISCreateGeneral(PETSC_COMM_WORLD, dm->nVetrices, glIndices, &oldIS);
		ISCreateStride(PETSC_COMM_WORLD, dm->nVetrices, dm->startIndex,1,&newIS);
		AOCreateBasicIS(oldIS, newIS, &(dm->procesOrdering));

		for (int i = 0; i < dm->nElements; i++) {
			MPI_Recv(&(dm->elements[i].numVetrices), 1, MPI_INT, 0, 0, PETSC_COMM_WORLD, &stats);
			MPI_Recv(&(dm->elements[i].vetrices), dm->elements[i].numVetrices, MPI_INT, 0, 0, PETSC_COMM_WORLD, &stats);
			AOApplicationToPetsc(dm->procesOrdering, dm->elements[i].numVetrices, dm->elements[i].vetrices);
		}

		while (true) {
			int borderId;
			MPI_Recv(&borderId, 1, MPI_INT, 0, 0, PETSC_COMM_WORLD, &stats);
			if (borderId == -1) break;
			AOApplicationToPetsc(dm->procesOrdering, 1, &borderId);
			dm->indDirchlet.insert(borderId);
		}

		MPI_Bcast(&(dm->nPairs), 1, MPI_INT, 0, PETSC_COMM_WORLD);
	}
}

void extractLocalAPart(Mat A, Mat *Aloc) {

	PetscInt m,n;
	MatGetOwnershipRange(A, &m, &n);
	PetscInt size = n - m;


	IS ISlocal;
	ISCreateStride(PETSC_COMM_SELF, size, m , 1, &ISlocal);
	Mat *sm;
	MatGetSubMatrices(A, 1, &ISlocal, &ISlocal, MAT_INITIAL_MATRIX, &sm);
	*Aloc = *sm;
}
