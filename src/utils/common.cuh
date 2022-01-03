#ifndef _COMMON_CUH
#define _COMMON_CUH

#include <cuda_runtime.h>
#include <stdio.h>

__host__ cudaError_t InitialCuda(int device);
__host__ cudaError_t ReleaseCuda(void);

#endif