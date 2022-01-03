#ifndef _SUM_CUDA_CUH
#define _SUM_CUDA_CUH

#include <cuda_runtime.h>
#include <stdio.h>
#include "utils/common.cuh"

__host__ void SumWithCuda(float* retValue, const float* data, size_t len);

#endif