#include "stdafx.h"
#include "Utilities.h"
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "cuda_runtime_api.h"

FILE* OpenFile(char* fileName, char* type)
{
	FILE* fp;
	fopen_s(&fp, fileName, type);

	if(!fp) 
	{
		fprintf(stderr, "ERROR: Could not open dat file %s\n", fileName);
		getchar();
		exit(-1);
	}

	return fp;
}

// Check for CUDA errors
void PrintError(cudaError_t err, char* file, int line) 
{
	
	if(err != cudaSuccess) 
	{
		fprintf(stderr, "CUDA ERROR: %s, file %s, line(%d)\n", cudaGetErrorString(err), file, line);
		getchar();
	}

}

// Check for Cublas errors
void CublasErrorCheck(cublasStatus_t err) 
{
	
	if(err != cudaSuccess) 
	{
		fprintf(stderr, "Cublas returned error code: %d, file %s, line(%d)\n", err, __FILE__, __LINE__);
		getchar();
	}

}

void PrintAvailableMemory() 
{

	size_t free_byte ;
    size_t total_byte ;
    cudaMemGetInfo( &free_byte, &total_byte ) ;

    double free_db = (double)free_byte ;
    double total_db = (double)total_byte ;
    double used_db = total_db - free_db ;

    printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
        used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);

}


void StartCudaTimer(float& time, cudaEvent_t& start, cudaEvent_t& stop)
{
	time = 0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
}

void StopCudaTimer(float& time, cudaEvent_t& start, cudaEvent_t& stop)
{
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
}
