#include <iostream>
#include <omp.h>
#include "sort.hpp"

using namespace std;

int main(int argc, char **argv)
{

  bool result = false;

  double begin_t = omp_get_wtime();
  InitData();
  double finish_t = omp_get_wtime();

  printf("init time consumption is %f s \r\n", finish_t - begin_t);

  begin_t = omp_get_wtime();
  Sort(rawFloatData, DATANUM);
  finish_t = omp_get_wtime();

  result = CheckSortResult(rawFloatData, DATANUM, true);
  printf("------------------------\r\n");
  printf("check result: %d\r\n", result);
  printf("Max() time consumption is %f s \r\n", finish_t - begin_t);

  begin_t = omp_get_wtime();
  SortSpeedUp(rawFloatData, DATANUM);
  finish_t = omp_get_wtime();

  result = CheckSortResult(rawFloatData, DATANUM, true);
  printf("------------------------\r\n");
  printf("check result: %d\r\n", result);
  printf("Max() time consumption is %f s \r\n", finish_t - begin_t);

  return 0;
}