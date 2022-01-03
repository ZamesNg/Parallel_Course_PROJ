#ifndef _SUM_HPP
#define _SUM_HPP

#include <stdio.h>
#include <math.h>

#include "sum_cuda.cuh"
#include "utils/common.hpp"

float Sum(const float data[],const size_t len); 
float SumSpeedUpOmp(const float data[], const size_t len);
float SumSpeedUpAvx(const float data[],const size_t len);
float SumSpeedUpAvxOmp(const float data[],const size_t len);

#endif