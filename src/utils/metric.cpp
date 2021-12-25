#include "metric.h"

double Clock2Second(time_t duration)
{
  return (double)(duration) / CLOCKS_PER_SEC;
}