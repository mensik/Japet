#include "japetLa.h"

void display(double *A, int m, int n) {
	using namespace std;
	cout << endl;
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			cout << A[i*n + j] << "\t";
		}
		cout << endl;
	}
	cout << endl;
}

/**
 @brief Extract submatrix by elimination Mth row and Nth column
 */
void getMinor(double *A, int n, int M, int N, double *minor) {

	int p = 0;
	for (int i = 0; i < n; i++) {
		if (i != M) {
			int q = 0;
			for (int j = 0; j < n; j++) {
				if (j != N) {
					minor[p * (n - 1) + q] = A[i * n + j];
					q++;
				}
			}
			p++;
		}
	}
}

double matrixDet(double *A, int n) {
	double det = 0;

	if (n == 1) {
		det = A[0];
	} else if (n == 2) {
		det = A[0] * A[3] - A[1] * A[2];
	} else {
		double *minor = new double[(n - 1) * (n - 1)];

		for (int i = 0; i < n; i++) {
			getMinor(A, n, 0, i, minor);
			det += pow(-1,i) * A[i] * matrixDet(minor, n - 1);
		}

		delete [] minor;
	}

	return det;
}

void matrixTranspose(double *A, int m, int n, double *At) {
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++) {
			At[j * m + i] = A[i * n + j];
		}
}

void matrixSetSub(double *A, int m, int n, int *idxRow, int rCount, int *idxCol, int cCount, double *subMatrix) {
	for (int i = 0; i < rCount; i++)
		for (int j = 0; j < cCount; j++) {
			A[idxRow[i] * n + idxCol[j]] = subMatrix[i * cCount + j];
		}
}

void matrixSum(double *A, double a, double *B, double b, double *C, int m, int n) {
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++)
			C[i * n + j] = a * A[i * n + j] + b * B[i* n + j];
}

void matrixMult(double *A, int mA, int nA, double *B, int mB, int nB, double *C) {
	for (int i = 0; i < mA; i++)
		for (int j = 0; j < nB; j++) {
			C[i * nB + j] = 0;
			for (int k = 0; k < nA; k++) {
				C[i * nB + j] += A[i * nA + k] * B[k * nB + j];
			}
		}
}
