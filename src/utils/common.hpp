#ifndef _COMMON_HPP
#define _COMMON_HPP

#include <stdio.h>
#include <math.h>


#define MAX_THREADS 64
#define SUBDATANUM 2000000
#define DATANUM (SUBDATANUM * MAX_THREADS)

extern float rawFloatData[DATANUM];

void InitData();

#endif