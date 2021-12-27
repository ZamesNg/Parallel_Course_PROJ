#ifndef _SORT_CUDA_CUH
#define _SORT_CUDA_CUH

#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>

__host__ cudaError_t InitialCuda(int device);
__host__ void SortWithCuda(float* data_host, size_t len, bool dir);
__host__ cudaError_t ReleaseCuda(void);


#endif