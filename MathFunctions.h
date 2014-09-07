#ifndef MATHFUNC_H
#define MATHFUNC_H

#include "Matrix.h"
#include <cublas_v2.h>

void MatMultAB(cublasHandle_t handle, Matrix<float>& A, Matrix<float>& B, Matrix<float>& C);
void MatMultATB(cublasHandle_t handle, Matrix<float>& A, Matrix<float>& B, Matrix<float>& C);
void MatMultABT(cublasHandle_t handle, Matrix<float>& A, Matrix<float>& B, Matrix<float>& C);
void SumRows(cublasHandle_t handle, Matrix<float>& A, Matrix<float>& B);
void ComputeNorm2(cublasHandle_t handle, Matrix<float>& A, float& res);

#endif
