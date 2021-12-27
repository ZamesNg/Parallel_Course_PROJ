#include <omp.h>
#include <immintrin.h>
#include <malloc.h>

#include "sum.hpp"

alignas(32) float rawFloatData[DATANUM];

void InitData()
{
  // #pragma omp parallel for
  for (size_t i = 0; i < DATANUM; ++i)
  {
    rawFloatData[i] = float(rand()) * 1e-15f;
    // rawFloatData[i] = i * 1e-5f;
  }
}

float Sum(const float data[], const size_t len)
{
  float sum_num_origin = 0.0f;
  float sum_num_read = 0.0f;
  float cur_num_read = 0.0f;

  for (size_t i = 0; i < len; ++i)
  {
    // will be optimize out?
    cur_num_read = sqrt(sqrt(data[i]));
    sum_num_read = sqrt(sqrt(sum_num_origin));
    cur_num_read += sum_num_read;

    sum_num_origin += data[i];
  }
  return sum_num_origin;
}

float SumSpeedUpOmp(const float data[], const size_t len)
{
  float sum_num_origin = 0.0f;
  float sum_num_read = 0.0f;
  float cur_num_read = 0.0f;

#pragma omp parallel for reduction(+ \
                                   : sum_num_origin) shared(data) private(cur_num_read, sum_num_read)
  for (size_t i = 0; i < len; ++i)
  {
    cur_num_read = sqrt(sqrt(data[i]));
    sum_num_read = sqrt(sqrt(sum_num_origin));
    cur_num_read += sum_num_read;

    sum_num_origin += data[i];
  }
  return sum_num_origin;
}

float SumSpeedUpAvx(const float data[], const size_t len)
{
  float sum = 0.0f;
  size_t num_iters = len / 8;
  int num_left = len - num_iters * 8;

  printf("iters: %d \t left:%d \r\n", num_iters, num_left);
  __m256 *ptr = (__m256 *)data;
  alignas(32) __m256 sum_num_origin = _mm256_set1_ps(0.0f);
  alignas(32) __m256 sum_num_read = _mm256_set1_ps(0.0f);
  alignas(32) __m256 cur_num_read = _mm256_set1_ps(0.0f);

  for (size_t i = 0; i < num_iters; ++i, ++ptr)
  {
    cur_num_read = _mm256_sqrt_ps(_mm256_sqrt_ps(*ptr));
    sum_num_read = _mm256_sqrt_ps(_mm256_sqrt_ps(sum_num_origin));

    sum_num_origin = _mm256_add_ps(sum_num_origin, *ptr);
  }

  // float *float_ptr = (float *)&sum_num_origin;
  float *float_ptr;
  float_ptr = (float *)memalign(32, 8 * sizeof(float));
  _mm256_store_ps(float_ptr, sum_num_origin);

  for (int i = 0; i < 8; i++)
  {
    sqrt(sqrt(sum));
    sqrt(sqrt(float_ptr[i]));
    sum += float_ptr[i];
  }

  for (int i = len - num_left; i < len; ++i)
  {
    sqrt(sqrt(sum));
    sqrt(sqrt(data[i]));
    sum += data[i];
  }

  return sum;
}


// User-Defined Reduction of OpenMP
// https://passlab.github.io/Examples/contents/Examples_udr.html
void _my_mm256_add_ps(__m256 *out, __m256 *in)
{
  // store the result of (out + in) to out
  *out = _mm256_add_ps(*out, *in);
}

#pragma omp declare reduction(sum_256:__m256                         \
                              : _my_mm256_add_ps(&omp_out, &omp_in)) \
    initializer(omp_priv = _mm256_set1_ps(0.0f))

float SumSpeedUpAvxOmp(const float data[], const size_t len)
{
  float sum = 0.0f;
  size_t num_iters = len / 8;
  int num_left = len - num_iters * 8;

  printf("iters: %d \t left:%d \r\n", num_iters, num_left);
  __m256 *ptr = (__m256 *)data;
  alignas(32) __m256 sum_num_origin = _mm256_set1_ps(0.0f);
  alignas(32) __m256 sum_num_read = _mm256_set1_ps(0.0f);
  alignas(32) __m256 cur_num_read = _mm256_set1_ps(0.0f);

#pragma omp parallel for reduction(sum_256 \
                                   : sum_num_origin) shared(ptr) private(cur_num_read, sum_num_read)
  for (size_t i = 0; i < num_iters; ++i)
  {
    cur_num_read = _mm256_sqrt_ps(_mm256_sqrt_ps(*ptr));
    sum_num_read = _mm256_sqrt_ps(_mm256_sqrt_ps(sum_num_origin));

    // sum_num_origin = _mm256_add_ps(sum_num_origin, *ptr);
    _my_mm256_add_ps(&sum_num_origin, ptr);
    ++ptr;
  }

  // float *float_ptr = (float *)&sum_num_origin;
  float *float_ptr;
  float_ptr = (float *)memalign(32, 8 * sizeof(float));
  _mm256_store_ps(float_ptr, sum_num_origin);

  for (int i = 0; i < 8; i++)
  {
    sqrt(sqrt(sum));
    sqrt(sqrt(float_ptr[i]));
    sum += float_ptr[i];
  }

  for (int i = len - num_left; i < len; ++i)
  {
    sqrt(sqrt(sum));
    sqrt(sqrt(data[i]));
    sum += data[i];
  }

  return sum;
}