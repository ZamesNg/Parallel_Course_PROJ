#include <float.h>

#include "sort_cuda.cuh"

// __host__ static size_t GetSmallestPowerOfTwoLargerThan(size_t n) {
//   size_t k = 1;
//   while (k < n) {
//     k = k << 1;
//   }
//   return k;
// }

__host__ cudaError_t InitialCuda(int device) {
  // 初始化CUDA设备, 线程级别!
  cudaError_t cudaStatus;

  // 清除遗留错误
  cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "\n[Error] last execution failed: %s!\n",
            cudaGetErrorString(cudaStatus));
  }

  // 确定CUDA设备, 默认只选中第一个设备
  cudaStatus = cudaSetDevice(device);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr,
            "\n[Error] cudaSetDevice failed!  Do you have a CUDA-capable GPU "
            "installed?\n");
  }

  return cudaStatus;
}

__host__ cudaError_t ReleaseCuda(void) {
  // 重置CUDA设备, 进程级别!
  cudaError_t cudaStatus = cudaDeviceReset();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "\n[Error] cudaDeviceReset failed!\n");
  }

  return cudaStatus;
}
__global__ void BitonicSortKernal(float* data, size_t step, size_t len,
                                  bool dir) {
  size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= len) return;

  float tmp;
  size_t i = tid % step;
  if (i < step / 2) {
    if ((data[tid] < data[tid + step - 2 * i - 1]) ^ dir) {
      tmp = data[tid];
      data[tid] = data[tid + step - 2 * i - 1];
      data[tid + step - 2 * i - 1] = tmp;
    }
  }

  __syncthreads();
}

__global__ void BitonicMergeKernal(float* data, size_t step, size_t len,
                                   bool dir) {
  size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= len) return;

  float tmp;
  size_t i = tid % (2 * step);
  if (i < step) {
    if ((data[tid] < data[tid + step]) ^ dir) {
      tmp = data[tid];
      data[tid] = data[tid + step];
      data[tid + step] = tmp;
    }
  }
  __syncthreads();
}

__global__ void InitKernal(float* ptr, size_t len, bool dir) {
  size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= len) return;
  ptr[tid] = dir ? FLT_MAX : -FLT_MAX;
}

__host__ void SortWithCuda(float* data_host, size_t len, bool dir) {
  int block_size = 1024;
  dim3 block(block_size, 1);
  size_t num = 1;
  while (num < len) {
    num <<= 1;
  }

  float* data_dev = NULL;

  // may cause large memory waste
  cudaMalloc((void**)&data_dev, num * sizeof(float));

  // init data according to dir
  dim3 init_grid((num - len - 1) / block.x + 1, 1);
  cudaDeviceSynchronize();
  InitKernal<<<init_grid, block>>>(data_dev + len, num - len, dir);
  cudaDeviceSynchronize();

  cudaMemcpy(data_dev, data_host, len * sizeof(float), cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();
  dim3 grid((num - 1) / block.x + 1, 1);
  for (size_t step = 1; step <= num; step <<= 1) {
    BitonicSortKernal<<<grid, block>>>(data_dev, step, num, dir);
    for (size_t s = step / 4; s > 0; s /= 2)
      BitonicMergeKernal<<<grid, block>>>(data_dev, s, num, dir);
  }
  cudaDeviceSynchronize();
  cudaMemcpy(data_host, data_dev, len * sizeof(float), cudaMemcpyDeviceToHost);
}