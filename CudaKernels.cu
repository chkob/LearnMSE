#include "CudaWrappers.cuh"

inline __device__ int sign(float x) 
{
	if(x < -EPSILON) 
	{
		return -1;
	} 
	else if (x > EPSILON) 
	{
		return 1;
	} 
	else 
	{
		return 0;
	}

}

// Sigmoid inline funcs
__device__ float sigmoidFunc(float x) { return 1.0 / (1.0 + __expf(-x)); }
__device__ float sigmoidDerivFunc(float x) { return x * (1.0 - x); } // here the x = activate(input)

// Rectified linear inline funcs
__device__ float rectifiedLinearFunc(float x) { return __logf(1.0 + __expf(x)); }
__device__ float rectifiedLinearDerivFunc(float x) { return (1.0 - __expf(-x)); }
	//1.0 / (1.0 + __expf(-x)); }

// Hyperbolic tangent inline funcs
__device__ float tanhFunc(float x) { return 1.7159f * (__expf(2.0f/3.0f * x) - __expf(-2.0f/3.0f * x)) / (__expf(2.0f/3.0f * x) + __expf(-2.0f/3.0f * x)); }
__device__ float tanhDerivFunc(float x) { return 2.0f/3.0f * (1.7159f - (x * x) / 1.7159f); }

// Linear inline funcs
__device__ float linearFunc(float x) { return x; };
__device__ float linearDerivFunc(float x) { return 1.0; };

// Tansig inline funcs
__device__ float tansigFunc(float x) { return (2.0 / (1.0 + __expf(-2.0*x)) - 1.0f); }
__device__ float tansigDerivFunc(float x) { return (1.0 - x*x); }


//__global__ void CudaAddBiasApplyActivation(ActivationFunction devFunc, float* a, float* b, int width, int height) 
//{
//	int xIndex = threadIdx.x + (blockIdx.x * blockDim.x);
//	int yIndex = threadIdx.y + (blockIdx.y * blockDim.y);
//	int index = yIndex * width + xIndex;
//	if(yIndex < height && xIndex < width) 
//		a[index] = (*devFunc)(a[index] + b[yIndex]);
//}

__global__ void CudaAddBiasApplyActivation(FUNC devFunc, float* a, float* b, int width, int height) 
{
	int xIndex = threadIdx.x + (blockIdx.x * blockDim.x);
	int yIndex = threadIdx.y + (blockIdx.y * blockDim.y);
	int index = yIndex * width + xIndex;
	if(yIndex < height && xIndex < width) {
		if (devFunc == TAN_SIG) {
			a[index] = tansigFunc(a[index] + b[yIndex]);
		}
		else if (devFunc == REC_LINEAR) {
			a[index] = rectifiedLinearFunc(a[index] + b[yIndex]);
		}
		else if (devFunc == LINEAR) {
			a[index] = linearFunc(a[index] + b[yIndex]);
		}
		else if (devFunc == SIGMOID) {
			a[index] = sigmoidFunc(a[index] + b[yIndex]);
		}
		//a[index] = (*devFunc)(a[index] + b[yIndex]);
	}
}

__global__ void CudaSubtractBFromA(float* a, float* b, float* c, int width, int height) 
{
	int xIndex = threadIdx.x + (blockIdx.x * blockDim.x);
	int yIndex = threadIdx.y + (blockIdx.y * blockDim.y);
	int index = yIndex * width + xIndex;
	if(yIndex < height && xIndex < width) 
		c[index] = a[index] - b[index];
}

__global__ void CudaSubtractBFromARelative(float* a, float* b, float* c, int width, int height) 
{
	int xIndex = threadIdx.x + (blockIdx.x * blockDim.x);
	int yIndex = threadIdx.y + (blockIdx.y * blockDim.y);
	int index = yIndex * width + xIndex;
	if(yIndex < height && xIndex < width) {
		c[index] = (a[index] - b[index]) / (0.01 + b[index]*b[index]);
	}
}

__global__ void CudaSubtractBFromARelative2(float* a, float* b, float* c, int width, int height) 
{
	int xIndex = threadIdx.x + (blockIdx.x * blockDim.x);
	int yIndex = threadIdx.y + (blockIdx.y * blockDim.y);
	int index = yIndex * width + xIndex;
	if(yIndex < height && xIndex < width) {
		c[index] = (a[index] - b[index]) / sqrt(0.01 + b[index]*b[index]);
	}
}

__global__ void CudaAddAToB(float* a, float* b, int size) 
{
	int tid = threadIdx.x + blockDim.x * threadIdx.y;
	int bid = blockIdx.x + gridDim.x * blockIdx.y;
	int index = tid + (blockDim.x * blockDim.y) * bid;
	if(index < size)  
		b[index] = a[index] + b[index];
}

//__global__ void CudaMultApplyActivationDeriv(ActivationFunction func, float* a, float* b, int width, int height) 
//{
//
//	int xIndex = threadIdx.x + (blockIdx.x * blockDim.x); 
//	int yIndex = threadIdx.y + (blockIdx.y * blockDim.y);
//	int index = yIndex * width + xIndex;
//	if(xIndex < width && yIndex < height)
//		a[index] = a[index] * (*func)(b[index]);
//}

__global__ void CudaMultApplyActivationDeriv(FUNC func, float* a, float* b, int width, int height) 
{

	int xIndex = threadIdx.x + (blockIdx.x * blockDim.x); 
	int yIndex = threadIdx.y + (blockIdx.y * blockDim.y);
	int index = yIndex * width + xIndex;
	if(xIndex < width && yIndex < height) {
		if (func == TAN_SIG_DERIV) {
			a[index] = a[index] * tansigDerivFunc(b[index]);
		}
		else if (func == REC_LINEAR_DERIV) {
			a[index] = a[index] * rectifiedLinearDerivFunc(b[index]);
		}
		else if (func == LINEAR_DERIV) {
			a[index] = a[index] * linearDerivFunc(b[index]);
		}
		else if (func == SIGMOID_DERIV) {
			a[index] = a[index] * sigmoidDerivFunc(b[index]);
		}
		//a[index] = a[index] * (*func)(b[index]);
	}
}

__global__ void CudaInitMatrix(float* a, const float val, int width, int height)
{
    int xIndex = threadIdx.x + (blockIdx.x * blockDim.x); 
	int yIndex = threadIdx.y + (blockIdx.y * blockDim.y);
	int index = yIndex + xIndex * height;
    if(xIndex < width && yIndex < height)
        a[index] = val;
}



__global__ void CudaRProp(float* currentWeights, float* currentGradients, float* meanSquaredGrad, float learningRate, int size, int iteration) 
{

	// Find row and col coordinate in grid
	int tid = threadIdx.x + blockDim.x * threadIdx.y;
	int bid = blockIdx.x + gridDim.x * blockIdx.y;
	int index = tid + (blockDim.x * blockDim.y) * bid;
	
	if (index < size)
	{
		if (iteration == 0)
		{
			meanSquaredGrad[index] = currentGradients[index] * currentGradients[index];
			currentWeights[index] -= learningRate * currentGradients[index] / sqrt(meanSquaredGrad[index]);	
		}
		else
		{
			meanSquaredGrad[index] = 0.9 * meanSquaredGrad[index] + 0.1 * currentGradients[index] * currentGradients[index];
			currentWeights[index] -= learningRate * currentGradients[index] / sqrt(meanSquaredGrad[index]);	
		}
	}
}

__global__ void CudaMultAByb(float* a, float b, int size) 
{
	int tid = threadIdx.x + blockDim.x * threadIdx.y;
	int bid = blockIdx.x + gridDim.x * blockIdx.y;
	int index = tid + (blockDim.x * blockDim.y) * bid;
	if(index < size) 
		a[index] = a[index] * b;
}
