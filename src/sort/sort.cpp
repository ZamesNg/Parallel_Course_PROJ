#include <iostream>
#include <omp.h>
#include "sort.hpp"
#include <sys/resource.h>
#include <memory.h>

using namespace std;

// declare of invisible function
void BitonicSortRecursionGeneral(float data[], size_t len, bool dir);
void BitonicSortRecursionParallel(float data[], size_t len, bool dir);
void BitonicSortNonRecursion(float data[], const size_t len, bool dir);

bool CheckSortResult(float *data, size_t len, bool dir)
{
  float tmp = 0;
  for (size_t i = 1; i < DATANUM; ++i)
  {
    tmp = data[i] - data[i - 1];
    if (((tmp > 0.0f) ^ dir) && tmp)
    {
      wcout << "find error at: " << i << " tmp: " << tmp << endl;
      return false;
    }
  }
  return true;
}

void Sort(float data[], const size_t len)
{
  BitonicSortRecursionGeneral(data, len, true);
}

void SortSpeedUp(float data[], const size_t len)
{
  BitonicSortNonRecursion(data, len, true);
}

// https://www.inf.hs-flensburg.de/lang/algorithmen/sortieren/bitonic/oddn.htm
size_t GetGreatestPowerOfTwoLessThan(size_t n)
{
  size_t k = 1;
  while (k < n)
  {
    k = k << 1;
  }
  return k >> 1;
}

void BitonicMergeRecursionGeneral(float data[], size_t len, bool dir)
{
  if (len > 1)
  {
    int step = GetGreatestPowerOfTwoLessThan(len);
    for (size_t i = 0; i < len - step; ++i)
    {
      if ((data[i] < data[i + step]) ^ dir)
      {
        // swap
        float tmp = data[i];
        data[i] = data[i + step];
        data[i + step] = tmp;
      }
    }
    BitonicMergeRecursionGeneral(data, step, dir);
    BitonicMergeRecursionGeneral(data + step, len - step, dir);
  }
}

void BitonicSortRecursionGeneral(float data[], size_t len, bool dir)
{
  if (len > 1)
  {
    size_t mid = len / 2;
    BitonicSortRecursionGeneral(data, mid, !dir);
    BitonicSortRecursionGeneral(data + mid, len - mid, dir);
    BitonicMergeRecursionGeneral(data, len, dir);
  }
}

void BitonicSortMergeNonRecursion(float data[], size_t len, bool dir)
{
  // dir = true merge up, false merge down
  size_t step = len / 2;
  while (step > 0)
  {
    for (size_t i = 0; i < len; i += step * 2)
    {
      for (size_t j = i; j < step + i; ++j)
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

void BitonicSortNonRecursion(float data[], const size_t len, bool dir)
{
  // will need padding
  for (size_t step = 2; step <= len / 2; step *= 2)
  {
    for (size_t i = 0; i < len; i += step * 2)
    {
      // printf("i:%ld \t step: %ld \r\n", i, step);
      BitonicSortMergeNonRecursion(data + i, step, !dir);
      BitonicSortMergeNonRecursion(data + i + step, step, dir);
    }
  }
  BitonicSortMergeNonRecursion(data, len, dir);
}

void MergeTwoSortedArray(float dataA[], size_t lenA, float dataB[], size_t lenB)
{
  int idx = lenA + lenB - 1;
  int idxA = lenA - 1;
  int idxB = lenB - 1;

  while (idx >= 0 && idxA >= 0 && idxB >= 0)
  {
    if (dataA[idxA] >= dataB[idxB])
    {
      dataA[idx] = dataA[idxA];
      idxA--;
    }
    else
    {
      dataA[idx] = dataB[idxB];
      idxB--;
    }
    idx--;
  }
  while (idxB >= 0)
  {
    dataA[idx] = dataB[idxB];
    idxB--;
    idx--;
  }
  wcout << "idx:" << idx << "\t idxA:" << idxA << "\t idxB:" << idxB << endl;
}