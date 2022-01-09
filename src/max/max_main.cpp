#include <omp.h>

#include "max.hpp"

int main(int argc, char **argv)
{

  float max = -1.0f;

  double begin_t = omp_get_wtime();
  InitData();
  double finish_t = omp_get_wtime();

  printf("init time consumption is %f s \r\n", finish_t - begin_t);

  printf("------------------------\r\n");
  for (int i; i < 5; i++)
  {
    begin_t = omp_get_wtime();
    max = Max(rawFloatData, DATANUM);
    finish_t = omp_get_wtime();

    printf("iter:%d. max number is %.2f. ", i, max);
    printf("Max() time consumption is %f s \r\n", finish_t - begin_t);
  }

  printf("------------------------\r\n");
  for (int i; i < 5; i++)
  {
    begin_t = omp_get_wtime();
    max = MaxSpeedUpOmp(rawFloatData, DATANUM);
    finish_t = omp_get_wtime();

    printf("iter:%d. max number is %.2f. ", i, max);
    printf("MaxSpeedUpOmp() time consumption is %f s \r\n", finish_t - begin_t);
  }

  printf("------------------------\r\n");
  for (int i; i < 5; i++)
  {
    begin_t = omp_get_wtime();
    max = MaxSpeedUpAvx(rawFloatData, DATANUM);
    finish_t = omp_get_wtime();

    printf("iter:%d. max number is %.2f. ", i, max);
    printf("MaxSpeedUpAvx() time consumption is %f s \r\n", finish_t - begin_t);
  }

  printf("------------------------\r\n");
  for (int i; i < 5; i++)
  {
    begin_t = omp_get_wtime();
    max = MaxSpeedUpAvxOmp(rawFloatData, DATANUM);
    finish_t = omp_get_wtime();

    printf("iter:%d. max number is %.2f. ", i, max);
    printf("MaxSpeedUpAvxOmp() time consumption is %f s \r\n", finish_t - begin_t);
  }

  
  printf("------------------------\r\n");
  for (int i; i < 5; i++)
  {
    InitialCuda(0);
    begin_t = omp_get_wtime();
    MaxWithCuda(&max, rawFloatData, DATANUM);
    finish_t = omp_get_wtime();

    printf("iter:%d. max number is %.2f. ", i, max);
    printf("MaxWithCuda() time consumption is %f s \r\n", finish_t - begin_t);
    ReleaseCuda();
  }
  
  return 0;
}