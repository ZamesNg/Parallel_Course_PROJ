#ifndef _MAX_CUDA_CUH
#define _MAX_CUDA_CUH

#include <cuda_runtime.h>
#include <stdio.h>

__host__ cudaError_t initialCuda(int device);
__host__ void maxWithCuda(float* retValue, const float* data, size_t len);
__host__ cudaError_t releaseCuda(void);

#endif