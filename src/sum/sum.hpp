#ifndef _SUM_HPP
#define _SUM_HPP

#include <stdio.h>
#include <math.h>

#define MAX_THREADS 64
#define SUBDATANUM 2000000
#define DATANUM (SUBDATANUM * MAX_THREADS)

extern float rawFloatData[DATANUM];

float Sum(const float data[],const size_t len); 
float SumSpeedUpOmp(const float data[], const size_t len);
float SumSpeedUpAvx(const float data[],const size_t len);
float SumSpeedUpAvxOmp(const float data[],const size_t len);

void InitData();

#endif