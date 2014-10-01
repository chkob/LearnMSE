#ifndef CUDAWRAPPERS_H
#define CUDAWRAPPERS_H

#include <cuda.h>
#include <cuda_runtime.h>
#include "Matrix.h"

typedef float (*ActivationFunction)(float);


__global__ void CudaAddBiasApplyActivation(FUNC devFunc, float* a, float* b, int width, int height);
__global__ void CudaSubtractBFromA(float* a, float* b, float* c, int width, int height);
__global__ void CudaSubtractBFromARelativeC(float* a, float* b, float* c, int width, int height, float cc);
__global__ void CudaMultApplyActivationDeriv(FUNC func, float* a, float* b, int width, int height);
__global__ void CudaInitMatrix(float* a, const float val, int width, int height);
__global__ void CudaRProp(float* currentWeights, float* currentGradients, float* meanSquaredGrad, float learningRate, int size, int iteration);
__global__ void CudaAddAToB(float* a, float* b, int size);
__global__ void CudaMultAByb(float* a, float b, int size);

void AddBiasApplyActivation(FUNC func, Matrix<float>& A, Matrix<float>& B) 
{
	assert(A.getWidth() < CUDA_MAX_NUM_BLOCKS * NUM_THREAD_BLOCK_X);
	assert(A.getHeight() < CUDA_MAX_NUM_BLOCKS);
	assert(A.getIsCudaMat() == true && B.getIsCudaMat() == true);
	assert(A.getHeight() == B.getHeight());
	assert(A.getDepth() == 1 && B.getDepth() == 1);

	// Initialize number of blocks and threads
	dim3 numOfThreadsPerBlock(NUM_THREAD_BLOCK_X, 1);

	int xSize = (A.getWidth() + numOfThreadsPerBlock.x - 1) / numOfThreadsPerBlock.x;
	int ySize = (A.getHeight() + numOfThreadsPerBlock.y - 1) / numOfThreadsPerBlock.y;

	xSize = pow(2, ceil( log(float(xSize)) / log(2.0f) ));
	ySize = pow(2, ceil( log(float(ySize)) / log(2.0f) ));

	dim3 numOfBlocks(xSize, ySize);

	//ActivationFunction* hostFunc = findActivationFunc(func);
	//ActivationFunction devFunc;
	//GpuErrorCheck(cudaMemcpyFromSymbol(&devFunc, *hostFunc, sizeof(ActivationFunction)));

	CudaAddBiasApplyActivation<<<numOfBlocks, numOfThreadsPerBlock>>>(func, A.getElements(), B.getElements(), A.getWidth(), A.getHeight());

	GpuErrorCheck(cudaPeekAtLastError());
	GpuErrorCheck(cudaDeviceSynchronize());
}

void SubtractBFromA(Matrix<float>& A, Matrix<float>& B, Matrix<float>& C) 
{
	assert(A.getWidth() < CUDA_MAX_NUM_BLOCKS * NUM_THREAD_BLOCK_X);
	assert(A.getHeight() < CUDA_MAX_NUM_BLOCKS);
	assert(A.getIsCudaMat() == true && B.getIsCudaMat() == true && C.getIsCudaMat() == true);
	assert(A.getHeight() == B.getHeight() && B.getHeight() == C.getHeight());
	assert(A.getWidth() == B.getWidth() && B.getWidth() == C.getWidth());
	assert(A.getDepth() == 1 &&  B.getDepth() == 1 && C.getDepth() == 1 );

	// Initialize number of blocks and threads
	dim3 numOfThreadsPerBlock(NUM_THREAD_BLOCK_X, 1);

	int xSize = (A.getWidth() + numOfThreadsPerBlock.x - 1) / numOfThreadsPerBlock.x;
	int ySize = (A.getHeight() + numOfThreadsPerBlock.y - 1) / numOfThreadsPerBlock.y;

	xSize = pow(2, ceil( log(float(xSize)) / log(2.0f) ));
	ySize = pow(2, ceil( log(float(ySize)) / log(2.0f) ));

	dim3 numOfBlocks(xSize, ySize);

	CudaSubtractBFromA<<<numOfBlocks, numOfThreadsPerBlock>>>(A.getElements(), B.getElements(), C.getElements(), A.getWidth(), A.getHeight());

	GpuErrorCheck(cudaPeekAtLastError());
	GpuErrorCheck(cudaDeviceSynchronize());
}

void SubtractBFromARelativeC(Matrix<float>& A, Matrix<float>& B, Matrix<float>& C, float c) 
{
	assert(A.getWidth() < CUDA_MAX_NUM_BLOCKS * NUM_THREAD_BLOCK_X);
	assert(A.getHeight() < CUDA_MAX_NUM_BLOCKS);
	assert(A.getIsCudaMat() == true && B.getIsCudaMat() == true && C.getIsCudaMat() == true);
	assert(A.getHeight() == B.getHeight() && B.getHeight() == C.getHeight());
	assert(A.getWidth() == B.getWidth() && B.getWidth() == C.getWidth());
	assert(A.getDepth() == 1 &&  B.getDepth() == 1 && C.getDepth() == 1 );

	// Initialize number of blocks and threads
	dim3 numOfThreadsPerBlock(NUM_THREAD_BLOCK_X, 1);

	int xSize = (A.getWidth() + numOfThreadsPerBlock.x - 1) / numOfThreadsPerBlock.x;
	int ySize = (A.getHeight() + numOfThreadsPerBlock.y - 1) / numOfThreadsPerBlock.y;

	xSize = pow(2, ceil( log(float(xSize)) / log(2.0f) ));
	ySize = pow(2, ceil( log(float(ySize)) / log(2.0f) ));

	dim3 numOfBlocks(xSize, ySize);

	CudaSubtractBFromARelativeC<<<numOfBlocks, numOfThreadsPerBlock>>>(A.getElements(), B.getElements(), C.getElements(), A.getWidth(), A.getHeight(), c);

	GpuErrorCheck(cudaPeekAtLastError());
	GpuErrorCheck(cudaDeviceSynchronize());
}

void MultApplyActivationDeriv(FUNC func, Matrix<float>& A, Matrix<float>& B) 
{
	assert(A.getWidth() < CUDA_MAX_NUM_BLOCKS * NUM_THREAD_BLOCK_X);
	assert(A.getHeight() < CUDA_MAX_NUM_BLOCKS);
	assert(A.getIsCudaMat() == true && B.getIsCudaMat() == true);
	assert(A.getHeight() == B.getHeight());
	assert(A.getDepth() == 1 && B.getDepth() == 1);

	// Initialize number of blocks and threads
	dim3 numOfThreadsPerBlock(NUM_THREAD_BLOCK_X, 1);

	int xSize = (A.getWidth() + numOfThreadsPerBlock.x - 1) / numOfThreadsPerBlock.x;
	int ySize = (A.getHeight() + numOfThreadsPerBlock.y - 1) / numOfThreadsPerBlock.y;

	xSize = pow(2, ceil( log(float(xSize)) / log(2.0f) ));
	ySize = pow(2, ceil( log(float(ySize)) / log(2.0f) ));

	dim3 numOfBlocks(xSize, ySize);

	/*ActivationFunction* hostFunc = findActivationFunc(func);
	ActivationFunction devFunc;
	GpuErrorCheck(cudaMemcpyFromSymbol(&devFunc, *hostFunc, sizeof(ActivationFunction)));*/

	CudaMultApplyActivationDeriv<<<numOfBlocks, numOfThreadsPerBlock>>>(func, A.getElements(), B.getElements(), A.getWidth(), A.getHeight());

	GpuErrorCheck(cudaPeekAtLastError());
	GpuErrorCheck(cudaDeviceSynchronize());
}

void InitMatrix(Matrix<float>& A, float val) 
{
	assert(A.getHeight() < CUDA_MAX_NUM_BLOCKS * NUM_THREAD_BLOCK_X);
	assert(A.getWidth() < CUDA_MAX_NUM_BLOCKS);
	assert(A.getIsCudaMat() == true);
	assert(A.getDepth() == 1);

	// Initialize number of blocks and threads
	dim3 numOfThreadsPerBlock(1, NUM_THREAD_BLOCK_X);

	int xSize = (A.getWidth() + numOfThreadsPerBlock.x - 1) / numOfThreadsPerBlock.x;
	int ySize = (A.getHeight() + numOfThreadsPerBlock.y - 1) / numOfThreadsPerBlock.y;

	xSize = pow(2, ceil( log(float(xSize)) / log(2.0f) ));
	ySize = pow(2, ceil( log(float(ySize)) / log(2.0f) ));

	dim3 numOfBlocks(xSize, ySize);

	CudaInitMatrix<<<numOfBlocks, numOfThreadsPerBlock>>>(A.getElements(), val, A.getWidth(), A.getHeight());

	GpuErrorCheck(cudaPeekAtLastError());
	GpuErrorCheck(cudaDeviceSynchronize());
}

void RProp(Matrix<float>& currentWeights, Matrix<float>& currentGradients, Matrix<float>& meanSquaredGrad, float learningRate, int iteration) {

	/*assert(currentWeights.getWidth() == currentGradients.getWidth() && currentWeights.getHeight() == currentGradients.getHeight());
	assert(currentWeights.getWidth() == localStepSizes.getWidth() && currentWeights.getHeight() == localStepSizes.getHeight());
	assert(currentWeights.getWidth() == previousGradients.getWidth() && currentWeights.getHeight() == previousGradients.getHeight());
	assert(currentWeights.getWidth() == previousUpdates.getWidth() && currentWeights.getHeight() == previousUpdates.getHeight());

	assert(currentWeights.getIsCudaMat());
	assert(currentGradients.getIsCudaMat());
	assert(localStepSizes.getIsCudaMat());
	assert(previousGradients.getIsCudaMat());
	assert(previousUpdates.getIsCudaMat());*/

	dim3 numOfThreadsPerBlock(NUM_THREAD_BLOCK_XY, NUM_THREAD_BLOCK_XY);
	int xSize = (currentWeights.getWidth() + numOfThreadsPerBlock.x - 1) / numOfThreadsPerBlock.x;
	int ySize = (currentWeights.getHeight() + numOfThreadsPerBlock.y - 1) / numOfThreadsPerBlock.y;
	dim3 numOfBlocks(xSize, ySize);

	CudaRProp<<<numOfBlocks, numOfThreadsPerBlock>>>(currentWeights.getElements(), currentGradients.getElements(), meanSquaredGrad.getElements(), learningRate, currentWeights.getWidth() * currentWeights.getHeight(), iteration);

	GpuErrorCheck(cudaPeekAtLastError());
	GpuErrorCheck(cudaDeviceSynchronize());

}

void MultAByb(Matrix<float>& A, float b) 
{
	assert(A.getIsCudaMat() == true);
	assert(A.getDepth() == 1);

	// Initialize number of blocks and threads
	dim3 numOfThreadsPerBlock(NUM_THREAD_BLOCK_XY, NUM_THREAD_BLOCK_XY);

	int xSize = (A.getWidth() + numOfThreadsPerBlock.x - 1) / numOfThreadsPerBlock.x;
	int ySize = (A.getHeight() + numOfThreadsPerBlock.y - 1) / numOfThreadsPerBlock.y;

	xSize = pow(2, ceil( log(float(xSize)) / log(2.0f) ));
	ySize = pow(2, ceil( log(float(ySize)) / log(2.0f) ));

	dim3 numOfBlocks(xSize, ySize);

	CudaMultAByb<<<numOfBlocks, numOfThreadsPerBlock>>>(A.getElements(), b, A.getWidth() * A.getHeight());

	GpuErrorCheck(cudaPeekAtLastError());
	GpuErrorCheck(cudaDeviceSynchronize());
}

void AddAToB(Matrix<float>& A, Matrix<float>& B) 
{
	assert(A.getWidth() < CUDA_MAX_NUM_BLOCKS * NUM_THREAD_BLOCK_X);
	assert(A.getHeight() < CUDA_MAX_NUM_BLOCKS);
	assert(A.getIsCudaMat() == true && B.getIsCudaMat() == true );
	assert(A.getHeight() == B.getHeight());
	assert(A.getWidth() == B.getWidth());
	assert(A.getDepth() == 1 &&  B.getDepth() == 1);

	// Initialize number of blocks and threads
	dim3 numOfThreadsPerBlock(NUM_THREAD_BLOCK_XY, NUM_THREAD_BLOCK_XY);

	int xSize = (A.getWidth() + numOfThreadsPerBlock.x - 1) / numOfThreadsPerBlock.x;
	int ySize = (A.getHeight() + numOfThreadsPerBlock.y - 1) / numOfThreadsPerBlock.y;

	xSize = pow(2, ceil( log(float(xSize)) / log(2.0f) ));
	ySize = pow(2, ceil( log(float(ySize)) / log(2.0f) ));

	dim3 numOfBlocks(xSize, ySize);

	CudaAddAToB<<<numOfBlocks, numOfThreadsPerBlock>>>(A.getElements(), B.getElements(), A.getWidth() * A.getHeight());

	GpuErrorCheck(cudaPeekAtLastError());
	GpuErrorCheck(cudaDeviceSynchronize());
}

#endif