#ifndef _SUM_CUDA_CUH
#define _SUM_CUDA_CUH

#include <cuda_runtime.h>
#include <stdio.h>

__host__ cudaError_t InitialCuda(int device);
__host__ void SumWithCuda(float* retValue, const float* data, size_t len);
__host__ cudaError_t ReleaseCuda(void);

#endif