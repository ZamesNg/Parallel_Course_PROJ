#ifndef _MAX_CUDA_CUH
#define _MAX_CUDA_CUH

#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>
#include "utils/common.cuh"

__host__ void MaxWithCuda(float* retValue, const float* data, size_t len);


#endif