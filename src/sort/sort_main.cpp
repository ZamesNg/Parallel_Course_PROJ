#include <iostream>
#include "sort.hpp"

using namespace std;

int main(int argc, char **argv)
{
  InitData();
  Sort(rawFloatData, DATANUM);

  bool result = CheckSortResult(rawFloatData, DATANUM, true);
  cout << "check result: " << result << endl;

  return 0;
}