#include <omp.h>
#include <immintrin.h>
#include "max.hpp"

alignas(32) float rawFloatData[DATANUM];

void InitData(const float data[], const size_t len)
{
  // #pragma omp parallel for
  for (size_t i = 0; i < len; ++i)
  {
    rawFloatData[i] = float(rand());
    // rawFloatData[i] = float(i + 1);
  }
}

float Max(const float data[], const size_t len)
{
  float max_num_origin = 1e-30f;
  float max_num_read = 0.0f;
  float cur_num_read = 0.0f;

  for (size_t i = 0; i < len; ++i)
  {
    cur_num_read = log(sqrt(rawFloatData[i]));
    max_num_read = log(sqrt(max_num_origin));

    if (cur_num_read >= max_num_read)
      max_num_origin = rawFloatData[i];
  }
  return max_num_origin;
}

float MaxSpeedUp(const float data[], const size_t len)
{
  size_t num_iters = len / 8;
  int num_left = len - num_iters * 8;

  printf("iters: %d \t left:%d \r\n", num_iters, num_left);
  __m256 *ptr = (__m256 *)data;
  alignas(32) __m256 max_num_origin = _mm256_set1_ps(1e-30f);
  alignas(32) __m256 max_num_read = _mm256_set1_ps(0.0f);
  alignas(32) __m256 cur_num_read = _mm256_set1_ps(0.0f);

  for (size_t i = 0; i < num_iters; ++i, ++ptr)
  {
    cur_num_read = _mm256_sqrt_ps(_mm256_sqrt_ps(*ptr));
    max_num_read = _mm256_sqrt_ps(_mm256_sqrt_ps(max_num_origin));

    max_num_origin = _mm256_max_ps(max_num_origin, *ptr);
  }

  float max = 1e-30f;
  float *float_ptr = (float *)&max_num_origin;

  for (int i = 0; i < 8; i++)
  {
    sqrt(sqrt(max));
    sqrt(sqrt(float_ptr[i]));
    if (max < float_ptr[i])
      max = float_ptr[i];
  }

  for (int i = len - num_left; i < len; i++)
  {
    sqrt(sqrt(max));
    sqrt(sqrt(data[i]));
    if (max < data[i])
      max = data[i];
  }
  return max;
}
