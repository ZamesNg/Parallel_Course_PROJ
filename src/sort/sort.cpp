#include <iostream>
#include "sort.hpp"

using namespace std;

float rawFloatData[DATANUM];

void InitData()
{
  // #pragma omp parallel for
  for (size_t i = 0; i < DATANUM; ++i)
  {
    rawFloatData[i] = float(rand());
    // rawFloatData[i] = float(int(i))*1e0f;
  }
}

bool CheckSortResult(float *data, size_t len, bool dir)
{
  float tmp = 0;
  for (size_t i = 1; i < DATANUM; ++i)
  {
    tmp = data[i] - data[i - 1];
    if (((tmp > 0.0f) ^ dir) && tmp)
    {
      cout << "find error at: " << i << " tmp: " << tmp << endl;
      return false;
    }
  }
  return true;
}

void BioSortMerge(float data[], size_t len, bool dir)
{
  // dir = true merge up, false merge down
  size_t step = len / 2;
  while (step > 0)
  {
    for (size_t i = 0; i < len; i += step * 2)
    {
      for (size_t j = i, k = 0; k < step; ++j, ++k)
      {
        if ((data[j] < data[j + step]) ^ dir)
        {
          // swap
          float tmp = data[j];
          data[j] = data[j + step];
          data[j + step] = tmp;
        }
      }
    }
    step /= 2;
  }
}

void Sort(float data[], const size_t len)
{

  for (size_t step = 2; step <= len / 2; step *= 2)
  {
    for (size_t i = 0; i < len; i += step * 2)
    {
      // printf("i:%ld \t step: %ld \r\n", i, step);
      BioSortMerge(data + i, step, true);
      BioSortMerge(data + i + step, step, false);
    }
  }
  BioSortMerge(data, len, true);
}