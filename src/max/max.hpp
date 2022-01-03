#ifndef _MAX_HPP
#define _MAX_HPP

#include <stdio.h>
#include <math.h>

#include "max_cuda.cuh"
#include "utils/common.hpp"

float Max(const float data[],const size_t len); 
float MaxSpeedUpOmp(const float data[], const size_t len);
float MaxSpeedUpAvx(const float data[],const size_t len);
float MaxSpeedUpAvxOmp(const float data[],const size_t len);

#endif