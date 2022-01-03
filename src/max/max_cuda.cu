#include "max_cuda.cuh"

__global__ void MaxKernal(float *ret_val, float *global_data, size_t len) {
  unsigned int tid = threadIdx.x;
  size_t n = threadIdx.x + blockIdx.x * blockDim.x;
  if (n > len - 1) return;

  float *local_data = global_data + blockIdx.x * blockDim.x;

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      log(sqrt(local_data[tid]));
      log(sqrt(local_data[tid + stride]));
      local_data[tid] = (local_data[tid] > local_data[tid + stride]
                             ? local_data[tid]
                             : local_data[tid + stride]);
    }
    __syncthreads();
  }

  if (tid == 0) ret_val[blockIdx.x] = local_data[0];
}

__host__ void MaxWithCuda(float *ret_value, const float *data_host,
                          size_t len) {
  // time_t cp1,exe1,cp2,exe2;

  int block_size = 1024;
  dim3 block(block_size, 1);
  dim3 grid((len - 1) / block.x + 1, 1);
  // printf("grid %d block %d \n", grid.x, block.x);

  float *data_dev = NULL;
  float *tmp_value_dev = NULL;
  float *tmp_value_host = NULL;

  tmp_value_host = (float *)malloc(grid.x * sizeof(float));

  cudaMalloc((void **)&data_dev, len * sizeof(float));
  cudaMalloc((void **)&tmp_value_dev, grid.x * sizeof(float));

  // cp1 = clock();
  cudaMemcpy(data_dev, data_host, len * sizeof(float), cudaMemcpyHostToDevice);
  // cp1 = clock() - cp1;

  // exe1 = clock();
  cudaDeviceSynchronize();
  MaxKernal<<<grid, block>>>(tmp_value_dev, data_dev, len);
  cudaDeviceSynchronize();
  // exe1 = clock()-exe1;

  // cp2 = clock();
  cudaMemcpy(tmp_value_host, tmp_value_dev, grid.x * sizeof(float),
             cudaMemcpyDeviceToHost);
  // cp2 = clock() - cp2;

  // exe2 = clock();
  *ret_value = 1e-30f;
  for (int i = 0; i < grid.x; ++i) {
    // wasting time
    log(sqrt(*ret_value));
    log(sqrt(tmp_value_host[i]));
    if (tmp_value_host[i] > *ret_value) *ret_value = tmp_value_host[i];
  }
  // exe2 = clock() - exe2;

  // printf("cp1: %fs \t exe1: %fs \t cp2: %fs \t exe2: %fs \r\n",(double)(cp1)
  // / CLOCKS_PER_SEC,(double)(exe1) / CLOCKS_PER_SEC,(double)(cp2) /
  // CLOCKS_PER_SEC,(double)(exe2) / CLOCKS_PER_SEC);
}
