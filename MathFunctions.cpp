#include "stdafx.h"
#include "MathFunctions.h"
#include "Utilities.h"
#include <assert.h>

void InitMatrix(Matrix<float>& A, float val);

void MatMultAB(cublasHandle_t handle, Matrix<float>& A, Matrix<float>& B, Matrix<float>& C)
{
	assert(A.getIsCudaMat() && B.getIsCudaMat() && C.getIsCudaMat());
	assert(A.getWidth() == B.getHeight());
	assert(A.getHeight() == C.getHeight() && B.getWidth() == C.getWidth());
	assert(A.getDepth() == 1 &&  B.getDepth() == 1 && C.getDepth() == 1 );

	float alpha = 1;
	float beta = 0;

	CublasErrorCheck(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, B.getWidth(), A.getHeight(), A.getWidth(), &alpha, B.getElements(), B.getWidth(), A.getElements(), A.getWidth(), &beta, C.getElements(), C.getWidth()));
}

void MatMultATB(cublasHandle_t handle, Matrix<float>& A, Matrix<float>& B, Matrix<float>& C)
{
	assert(A.getIsCudaMat() && B.getIsCudaMat() && C.getIsCudaMat());
	assert(A.getHeight() == B.getHeight());
	assert(A.getWidth() == C.getHeight() && B.getWidth() == C.getWidth());
	assert(A.getDepth() == 1 &&  B.getDepth() == 1 && C.getDepth() == 1 );

	float alpha = 1;
	float beta = 0;

	CublasErrorCheck(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, B.getWidth(), A.getWidth(), A.getHeight(), &alpha, B.getElements(), B.getWidth(), A.getElements(), A.getWidth(), &beta, C.getElements(), C.getWidth()));
}

void MatMultABT(cublasHandle_t handle, Matrix<float>& A, Matrix<float>& B, Matrix<float>& C)
{
	assert(A.getIsCudaMat() && B.getIsCudaMat() && C.getIsCudaMat());
	assert(A.getWidth() == B.getWidth());
	assert(A.getHeight() == C.getHeight() && B.getHeight() == C.getWidth());
	assert(A.getDepth() == 1 &&  B.getDepth() == 1 && C.getDepth() == 1 );

	float alpha = 1;
	float beta = 0;

	CublasErrorCheck(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, B.getHeight(), A.getHeight(), A.getWidth(), &alpha, B.getElements(), B.getWidth(), A.getElements(), A.getWidth(), &beta, C.getElements(), C.getWidth()));
}

void ComputeNorm2(cublasHandle_t handle, Matrix<float>& A, float& res)
{
	assert(A.getIsCudaMat());

	CublasErrorCheck(cublasSnrm2(handle, A.getWidth()*A.getHeight(), A.getElements(), 1, &res));
}

void SumRows(cublasHandle_t handle, Matrix<float>& A, Matrix<float>& C)
{
	assert(A.getIsCudaMat() && C.getIsCudaMat());
	assert(A.getHeight() == C.getHeight());
	assert(A.getDepth() == 1 &&  C.getDepth() == 1 && C.getWidth() == 1);

	Matrix<float> B(1, A.getWidth());
	B.AllocateData(true);
	InitMatrix(B, 1.0f);


	float alpha = 1;
	float beta = 0;

	CublasErrorCheck(cublasSgemv(handle, CUBLAS_OP_T, A.getWidth(), A.getHeight(), &alpha, A.getElements(), A.getWidth(), B.getElements(), 1, &beta, C.getElements(), 1));
}
