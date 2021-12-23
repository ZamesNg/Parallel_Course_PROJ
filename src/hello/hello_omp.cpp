#include <iostream>
#include <omp.h>

int main(void)
{

// here to check if we are using openmp
#if _OPENMP
  std::cout << " support openmp " << std::endl;
#else
  std::cout << " not support openmp" << std::endl;
#endif

// here is the code that should be parallel running
#pragma omp parallel for
  for (int i = 0; i < 10; i++)
  {
    std::cout << "Test" << i << std::endl;
  }

  return 0;
}