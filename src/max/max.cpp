#include <omp.h>
#include <immintrin.h>
#include <malloc.h>
#include "max.hpp"

float Max(const float data[], const size_t len)
{
  float max_num_origin = 0.0f;
  float max_num_read = 0.0f;
  float cur_num_read = 0.0f;

  for (size_t i = 0; i < len; ++i)
  {
    // will be optimize out?
    cur_num_read = sqrt(sqrt(data[i]));
    max_num_read = sqrt(sqrt(max_num_origin));

    if (max_num_origin < data[i])
      max_num_origin = data[i];
  }
  return max_num_origin;
}

float MaxSpeedUpOmp(const float data[], const size_t len)
{
  float max_num_origin = 0.0f;
  float max_num_read = 0.0f;
  float cur_num_read = 0.0f;

#pragma omp parallel for reduction(max \
                                   : max_num_origin) shared(data) private(cur_num_read, max_num_read)
  for (size_t i = 0; i < len; ++i)
  {
    cur_num_read = sqrt(sqrt(data[i]));
    max_num_read = sqrt(sqrt(max_num_origin));

    if (max_num_origin < data[i])
      max_num_origin = data[i];
  }
  return max_num_origin;
}

float MaxSpeedUpAvx(const float data[], const size_t len)
{
  float max = 0.0f;
  size_t num_iters = len / 8;
  int num_left = len - num_iters * 8;

  // printf("iters: %ld \t left:%d \r\n", num_iters, num_left);
  __m256 *ptr = (__m256 *)data;
  alignas(32) __m256 max_num_origin = _mm256_set1_ps(0.0f);
  alignas(32) __m256 max_num_read = _mm256_set1_ps(0.0f);
  alignas(32) __m256 cur_num_read = _mm256_set1_ps(0.0f);

  for (size_t i = 0; i < num_iters; ++i, ++ptr)
  {
    cur_num_read = _mm256_sqrt_ps(_mm256_sqrt_ps(*ptr));
    max_num_read = _mm256_sqrt_ps(_mm256_sqrt_ps(max_num_origin));

    max_num_origin = _mm256_max_ps(max_num_origin, *ptr);
  }

  // float *float_ptr = (float *)&max_num_origin;
  float *float_ptr;
  float_ptr = (float *)memalign(32, 8 * sizeof(float));
  _mm256_store_ps(float_ptr, max_num_origin);

  for (int i = 0; i < 8; i++)
  {
    sqrt(sqrt(max));
    sqrt(sqrt(float_ptr[i]));

    if (max < float_ptr[i])
      max = float_ptr[i];
  }

  for (int i = len - num_left; i < len; ++i)
  {
    sqrt(sqrt(max));
    sqrt(sqrt(data[i]));

    if (max < float_ptr[i])
      max = float_ptr[i];
  }

  return max;
}

// User-Defined Reduction of OpenMP
// https://passlab.github.io/Examples/contents/Examples_udr.html
void _my_mm256_max_ps(__m256 *out, __m256 *in)
{
  // store the result of (out + in) to out
  *out = _mm256_max_ps(*out, *in);
}

#pragma omp declare reduction(max_256:__m256                         \
                              : _my_mm256_max_ps(&omp_out, &omp_in)) \
    initializer(omp_priv = _mm256_set1_ps(0.0f))

float MaxSpeedUpAvxOmp(const float data[], const size_t len)
{
  float max = 0.0f;
  size_t num_iters = len / 8;
  int num_left = len - num_iters * 8;

  // printf("iters: %ld \t left:%d \r\n", num_iters, num_left);
  __m256 *ptr = (__m256 *)data;
  alignas(32) __m256 max_num_origin = _mm256_set1_ps(0.0f);
  alignas(32) __m256 max_num_read = _mm256_set1_ps(0.0f);
  alignas(32) __m256 cur_num_read = _mm256_set1_ps(0.0f);

#pragma omp parallel for reduction(max_256 \
                                   : max_num_origin) firstprivate(ptr, cur_num_read, max_num_read)
  for (size_t i = 0; i < num_iters; ++i)
  {
    cur_num_read = _mm256_sqrt_ps(_mm256_sqrt_ps(*ptr));
    max_num_read = _mm256_sqrt_ps(_mm256_sqrt_ps(max_num_origin));

    // max_num_origin = _mm256_max_ps(max_num_origin, *ptr);
    _my_mm256_max_ps(&max_num_origin, ptr);
    ++ptr;
  }

  // float *float_ptr = (float *)&max_num_origin;
  float *float_ptr;
  float_ptr = (float *)memalign(32, 8 * sizeof(float));
  _mm256_store_ps(float_ptr, max_num_origin);

  for (int i = 0; i < 8; i++)
  {
    sqrt(sqrt(max));
    sqrt(sqrt(float_ptr[i]));
    if (max < float_ptr[i])
      max = float_ptr[i];
  }

  for (int i = len - num_left; i < len; ++i)
  {
    sqrt(sqrt(max));
    sqrt(sqrt(data[i]));

    if (max < float_ptr[i])
      max = float_ptr[i];
  }

  return max;
}