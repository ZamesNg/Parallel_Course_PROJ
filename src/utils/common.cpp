#include "common.hpp"

alignas(32) float rawFloatData[DATANUM];

void InitData()
{
  // #pragma omp parallel for
  for (size_t i = 0; i < DATANUM; ++i)
  {
    rawFloatData[i] = float(rand()) * 1e-6f;
    // rawFloatData[i] = i * 1e-5f;
  }
}