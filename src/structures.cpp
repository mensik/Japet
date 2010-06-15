#include "structures.h"
 
void Mesh::dumpForMatlab(PetscViewer v) {
	Mat x;
	Mat e;

	PetscInt rank;
	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

	if (!rank) {
		PetscInt indexes[] = {0,1,2,3};
		PetscInt numVetrices = vetrices.size();

		MatCreateMPIAIJ(PETSC_COMM_WORLD, numVetrices, PETSC_DECIDE, PETSC_DECIDE, 4, 4,PETSC_NULL, 4,PETSC_NULL, &x);
		
		std::map<PetscInt, Point>::iterator point = vetrices.begin();
		for (int i = 0; i < numVetrices;i++) {
			PetscInt rowNumber = point->first;
			PetscScalar	data[] = {point->second.x, point->second.y, point->second.z, rank};
			MatSetValues(x,1, &rowNumber,4, indexes ,data,INSERT_VALUES);
			point++;
		}
		MatAssemblyBegin(x,MAT_FINAL_ASSEMBLY);
		PetscInt numElements = elements.size();
		MatCreateMPIAIJ(PETSC_COMM_WORLD, numElements, PETSC_DECIDE, PETSC_DECIDE, 4,4,PETSC_NULL, 4, PETSC_NULL, &e);	
		
		std::map<PetscInt, Element>::iterator el = elements.begin();
		for (int i = 0; i < numElements;i++) {
			PetscInt rowNumber = el->first;
			PetscScalar data[4];
			PetscInt counter = 0;
			for (int j = 0; j < el->second.numVetrices; j++)
				data[counter++] = el->second.vetrices[j];
			data[3] = rank;
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

				Point node(xPos,yPos);
				vetrices.insert(std::pair<PetscInt, Point>(nodeIndex, node));
				
				//Element creation
				if (j < yPoints - 1 && i < xPoints - 1) {
					Element el1, el2;

					el1.numVetrices = 3;
					el1.numEdges = 0;
					el1.vetrices[0] = nodeIndex;
					el1.vetrices[1] = nodeIndex + xPoints + 1;
					el1.vetrices[2] = nodeIndex + 1;
					
					el2.numVetrices = 3;
					el2.numEdges = 0;
					el2.vetrices[0] = nodeIndex;
					el2.vetrices[1] = nodeIndex + xPoints;
					el2.vetrices[2] = nodeIndex + xPoints +	 1;

					elements.insert(std::pair<PetscInt, Element>(elementIndex++, el1));
					elements.insert(std::pair<PetscInt, Element>(elementIndex++, el2));
				}
				nodeIndex++;
		}
	regenerateEdges();
	findBorders();
}

void Mesh::regenerateEdges() {
	PetscInt edgeIndex = 0;
	for (std::map<PetscInt, Element>::iterator i = elements.begin(); i != elements.end(); i++) {
		for (PetscInt j = 0; j < i->second.numVetrices; j++) {
			PetscInt v1 = i->second.vetrices[j];
			PetscInt v2 = i->second.vetrices[(j+1) % (i->second.numVetrices)];
			
			PetscInt edgeInd = getEdge(v1,v2);
			if (edgeInd == -1) { // Edge doesn't exist yet
				Edge newEdge;
				edgeInd = edgeIndex++;
				newEdge.vetrices[0] = v1;
				newEdge.vetrices[1] = v2;
				newEdge.elements.insert(i->first);
				vetrices[v1].edges.insert(edgeInd);
				vetrices[v2].edges.insert(edgeInd);
				edges.insert(std::pair<PetscInt, Edge>(edgeInd, newEdge));
			}
			i->second.edges[i->second.numEdges] = edgeInd;
			edges[edgeInd].elements.insert(i->first);
		}
	}
}

void Mesh::findBorders() {
	for (std::map<PetscInt, Edge>::iterator i = edges.begin(); i != edges.end(); i++) {
		if (i->second.elements.size() == 1) borderEdges.insert(i->first);
	}
}

PetscInt Mesh::getEdge(PetscInt nodeA, PetscInt nodeB) {

	PetscInt resultIndex = -1;
	for (std::set<PetscInt>::iterator i = vetrices[nodeA].edges.begin(); i != vetrices[nodeA].edges.end(); i++) {
		Edge e = edges[*i];
		if (e.vetrices[0] == nodeB || e.vetrices[1] == nodeB) {
			resultIndex = *i;
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
			for (std::map<PetscInt, Point>::iterator i = vetrices.begin(); i != vetrices.end(); i++)
				fprintf(f, "%d:%lf %lf %lf\n", i->first, i->second.x, i->second.y, i->second.z);
			fprintf(f, "Elements (id:numVetrices [vetrices])\n");
			fprintf(f, "Num: %d\n", (int)elements.size());
			for (std::map<PetscInt, Element>::iterator i = elements.begin(); i != elements.end(); i++) {
				fprintf(f, "%d:%d", i->first, i->second.numVetrices);
				for (int j = 0; j < i->second.numVetrices; j++)
					fprintf(f, " %d", i->second.vetrices[j]);
				fprintf(f, "\n");
			}
			if (withEdges) {
				fprintf(f, "Edges (id:vetriceA vetriceB)\n");
				fprintf(f, "Num: %d\n", (int)edges.size());
				for (std::map<PetscInt, Edge>::iterator i = edges.begin(); i != edges.end(); i++)
					fprintf(f,"%d:%d %d\n", i->first, i->second.vetrices[0], i->second.vetrices[1]);
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
				Point newPoint(x,y,z);
				vetrices.insert(std::pair<PetscInt, Point>(id, newPoint));
			}
			PetscInt numElements;
			fgets(msg, 128, f);
			fscanf(f, "Num: %d\n", &numElements);
			for (int i = 0; i < numElements; i++) {
				PetscInt id, size;
				fscanf(f, "%d:%d",&id, &size);
				Element newElement;
				newElement.numVetrices = size;
				for (int j = 0; j < size-1; j++)
					fscanf(f," %d",newElement.vetrices + j);
				fscanf(f," %d\n", newElement.vetrices + size-1);
				elements.insert(std::pair<PetscInt, Element>(id, newElement));	
			}
			if (withEdges) {
				PetscInt numEdges;
				fgets(msg, 128, f);
				fscanf(f, "Num: %d\n", &numEdges);
				for (int i = 0; i < numEdges; i++) {
					PetscInt id;
					Edge newEdge;
					fscanf(f, "%d: %d %d\n", &id, newEdge.vetrices, newEdge.vetrices + 1);
					
					vetrices[newEdge.vetrices[0]].edges.insert(id);
					vetrices[newEdge.vetrices[1]].edges.insert(id);
					
					edges.insert(std::pair<PetscInt, Edge>(id, newEdge));
				}
				for (std::map<PetscInt, Element>::iterator i = elements.begin(); i != elements.end(); i++) {
					for (PetscInt j = 0; j < i->second.numVetrices; j++) {
						PetscInt v1 = i->second.vetrices[j];
						PetscInt v2 = i->second.vetrices[(j+1) % (i->second.numVetrices)];
			
						PetscInt edgeInd = getEdge(v1,v2);
						i->second.edges[i->second.numEdges] = edgeInd;
						edges[edgeInd].elements.insert(i->first);
					}
				}
			}
			fclose(f);
		}
	}
}


void extractLocalAPart(Mat A, std::set<PetscInt> vetrices, Mat *Aloc) {
	PetscInt localIndexes[vetrices.size()];
	PetscInt counter = 0;
	for (std::set<PetscInt>::iterator i = vetrices.begin();
		i != vetrices.end(); i++) {
		localIndexes[counter++] = *i;
	}
	IS ISlocal;
	ISCreateGeneral(PETSC_COMM_SELF, vetrices.size(), localIndexes, &ISlocal);
	Mat *sm;
	MatGetSubMatrices(A, 1, &ISlocal, &ISlocal, MAT_INITIAL_MATRIX, &sm);
	*Aloc = *sm;
}

