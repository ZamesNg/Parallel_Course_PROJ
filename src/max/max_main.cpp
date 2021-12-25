#include <time.h>

#include "max.hpp"
#include "max_cuda.cuh"

int main(int argc, char **argv)
{

  float max = -1.0f;

  time_t begin_t = clock();
  InitData(rawFloatData, DATANUM);
  time_t finish_t = clock();

  printf("init time consumption is %f s \r\n", (double)(finish_t - begin_t) / CLOCKS_PER_SEC);

  begin_t = clock();
  max = MaxSpeedUp(rawFloatData, DATANUM);
  finish_t = clock();

  printf("max number is %.2f \r\n", max);
  printf("last number is %.2f \r\n", rawFloatData[DATANUM - 1]);
  printf("max() time consumption is %f s \r\n", (double)(finish_t - begin_t) / CLOCKS_PER_SEC);

  initialCuda(0);

  begin_t = clock();
  maxWithCuda(&max, rawFloatData, DATANUM);
  finish_t = clock();
  printf("max number is %.2f \r\n", max);
  printf("last number is %.2f \r\n", rawFloatData[DATANUM - 1]);
  printf("MaxWithCuda() time consumption is %f s \r\n", (double)(finish_t - begin_t) / CLOCKS_PER_SEC);

  return 0;
}