#ifndef _SORT_HPP
#define _SORT_HPP

#include <stdio.h>
#include <math.h>
#include "sort_cuda.cuh"
#include "utils/common.hpp"

void Sort(float data[], const size_t len);
void SortSpeedUp(float data[], const size_t len);
void MergeTwoSortedArray(float dataA[],size_t lenA,float dataB[],size_t lenB);

bool CheckSortResult(float *data, size_t len, bool dir);

#endif