#include <omp.h>

#include "max.hpp"

float rawFloatData[DATANUM];

void InitData(const float data[], const size_t len)
{
// #pragma omp parallel for
  for (size_t i = 0; i < len; ++i)
  {
    rawFloatData[i] = float(rand());
    // rawFloatData[i] = float(i + 1);
  }
}

float Max(const float data[], const int len)
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