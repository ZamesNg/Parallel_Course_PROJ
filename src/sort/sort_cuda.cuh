#ifndef _SORT_CUDA_CUH
#define _SORT_CUDA_CUH

#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>
#include "utils/common.cuh"

__host__ void SortWithCuda(float* data_host, size_t len, bool dir);

#endif