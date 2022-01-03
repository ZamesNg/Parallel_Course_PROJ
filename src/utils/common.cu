#include "common.cuh"

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