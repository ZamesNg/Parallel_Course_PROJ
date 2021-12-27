#include "sum_cuda.cuh"

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

__global__ void SumKernal(float *ret_val, float *global_data, size_t len) {
  unsigned int tid = threadIdx.x;
  size_t n = threadIdx.x + blockIdx.x * blockDim.x;
  if (n > len - 1) return;

  float *local_data = global_data + blockIdx.x * blockDim.x;

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sqrt(sqrt(local_data[tid]));
      sqrt(sqrt(local_data[tid + stride]));
      local_data[tid] = local_data[tid] + local_data[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) ret_val[blockIdx.x] = local_data[0];
}

__host__ void SumWithCuda(float *ret_value, const float *data_host,
                          size_t len) {
  int block_size = 1024;
  dim3 block(block_size, 1);
  dim3 grid((len - 1) / block.x + 1, 1);
  printf("grid %d block %d \n", grid.x, block.x);

  float *data_dev = NULL;
  float *tmp_value_dev = NULL;
  float *tmp_value_host = NULL;

  tmp_value_host = (float *)malloc(grid.x * sizeof(float));

  cudaMalloc((void **)&data_dev, len * sizeof(float));
  cudaMalloc((void **)&tmp_value_dev, grid.x * sizeof(float));

  cudaMemcpy(data_dev, data_host, len * sizeof(float), cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();
  SumKernal<<<grid, block>>>(tmp_value_dev, data_dev, len);
  cudaDeviceSynchronize();

  cudaMemcpy(tmp_value_host, tmp_value_dev, grid.x * sizeof(float),
             cudaMemcpyDeviceToHost);

  float _value = 0;
  for (int i = 0; i < grid.x; ++i) {
    // wasting time
    sqrt(sqrt(_value));
    sqrt(sqrt(tmp_value_host[i]));
    _value += tmp_value_host[i];
  }

  free(tmp_value_host);
  *ret_value = _value;
}

__host__ cudaError_t ReleaseCuda(void) {
  // 重置CUDA设备, 进程级别!
  cudaError_t cudaStatus = cudaDeviceReset();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "\n[Error] cudaDeviceReset failed!\n");
  }

  return cudaStatus;
}