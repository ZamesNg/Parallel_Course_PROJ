#ifndef _SORT_HPP
#define _SORT_HPP

#include <stdio.h>
#include <math.h>

#define MAX_THREADS 64
#define SUBDATANUM 4096
#define DATANUM (SUBDATANUM * MAX_THREADS) /*这个数值是总数据量*/

extern float rawFloatData[DATANUM];

void Sort(float data[], const size_t len);
void SortSpeedUp(float data[], const size_t len);

void InitData();
bool CheckSortResult(float *data, size_t len, bool dir);

#endif