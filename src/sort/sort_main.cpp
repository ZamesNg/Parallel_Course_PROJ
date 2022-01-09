#include <iostream>
#include <omp.h>
#include <string.h>

#include "sort.hpp"

using namespace std;

int main(int argc, char **argv)
{

  bool result = false;

  double begin_t = omp_get_wtime();
  InitData();
  double finish_t = omp_get_wtime();

  printf("init time consumption is %f s \r\n", finish_t - begin_t);

  float *tmp = new float[DATANUM];
  printf("------------------------\r\n");
  for (int i; i < 5; i++)
  {
    memcpy(tmp, rawFloatData, DATANUM * sizeof(float));
    begin_t = omp_get_wtime();
    Sort(tmp, DATANUM);
    finish_t = omp_get_wtime();

    result = CheckSortResult(tmp, DATANUM, true);
    printf("iter:%d. check result: %d. ", i, result);
    printf("Sort() time consumption is %f s \r\n", finish_t - begin_t);
  }

  printf("------------------------\r\n");
  for (int i; i < 5; i++)
  {
    memcpy(tmp, rawFloatData, DATANUM * sizeof(float));
    InitialCuda(0);
    begin_t = omp_get_wtime();
    SortWithCuda(tmp, DATANUM, true);
    finish_t = omp_get_wtime();

    result = CheckSortResult(tmp, DATANUM, true);
    printf("iter:%d. check result: %d. ", i, result);
    printf("SortWithCuda() time consumption is %f s \r\n", finish_t - begin_t);
    ReleaseCuda();
  }

  return 0;
}