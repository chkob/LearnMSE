#ifndef UTILITIES_H
#define UTILITIES_H

#include <stdio.h>
#include <cublas_v2.h>


FILE* OpenFile(char* fileName, char* type);
void PrintError(cudaError_t err, char* file, int line);
void CublasErrorCheck(cublasStatus_t err);
void PrintAvailableMemory();
void StartCudaTimer(float& time, cudaEvent_t& start, cudaEvent_t& stop);
void StopCudaTimer(float& time, cudaEvent_t& start, cudaEvent_t& stop);

#endif
