#ifndef JAPETLA_H_
#define JAPETLA_H_

#include <math.h>
#include <iostream>

void display(double *A, int m, int n);
void getMinor(double *A, int n, int M, int N, double *minor);
double matrixDet(double *A, int n);
void matrixSetSub(double *A, int m, int n, int *idxRow, int rCount, int *idxCol, int cCount, double *subMatrix);
void matrixTranspose(double *A, int m, int n, double *At);
void matrixSum(double *A, double a, double *B, double b, double *C, int m, int n);
void matrixMult(double *A, int mA, int nA, double *B, int mB, int nB, double *C);

#endif
