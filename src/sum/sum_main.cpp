#include <time.h>
#include <omp.h>

#include "sum.hpp"

int main(int argc, char **argv)
{

  float sum = -1.0f;

  double begin_t = omp_get_wtime();
  InitData();
  double finish_t = omp_get_wtime();

  printf("init time consumption is %f s \r\n", finish_t - begin_t);
  printf("sum should be %f \r\n", (rawFloatData[0] + rawFloatData[DATANUM - 1]) * DATANUM / 2);

  begin_t = omp_get_wtime();
  sum = Sum(rawFloatData, DATANUM);
  finish_t = omp_get_wtime();

  printf("sum number is %f \r\n", sum);
  printf("Sum() time consumption is %f s \r\n", finish_t - begin_t);

  begin_t = omp_get_wtime();
  sum = SumSpeedUpOmp(rawFloatData, DATANUM);
  finish_t = omp_get_wtime();

  printf("sum number is %f \r\n", sum);
  // printf("SumSpeedUpOmp() time consumption is %f s \r\n", finish_t - begin_t);
  printf("SumSpeedUpOmp() time consumption is %f s \r\n", finish_t - begin_t);

  begin_t = omp_get_wtime();
  sum = SumSpeedUpAvx(rawFloatData, DATANUM);
  finish_t = omp_get_wtime();

  printf("sum number is %f \r\n", sum);
  printf("SumSpeedUpAvx() time consumption is %f s \r\n", finish_t - begin_t);

  begin_t = omp_get_wtime();
  sum = SumSpeedUpAvxOmp(rawFloatData, DATANUM);
  finish_t = omp_get_wtime();

  printf("sum number is %f \r\n", sum);
  printf("SumSpeedUpAvxOmp() time consumption is %f s \r\n", finish_t - begin_t);

  InitialCuda(0);

  begin_t = omp_get_wtime();
  SumWithCuda(&sum, rawFloatData, DATANUM);
  finish_t = omp_get_wtime();
  printf("sum number is %f \r\n", sum);
  printf("SumWithCuda() time consumption is %f s \r\n", finish_t - begin_t);

  ReleaseCuda();

  return 0;
}