#include <iostream>

#include "mystring.hpp"
using namespace std;

int main(int argc, char **argv)
{
  MyString temp;
  MyString temp1(-567);
  MyString temp2(-5.67f);

  char buffer[10] = "abcd";
  MyString temp3(buffer);
  MyString temp4(temp1);

  temp = temp + temp2;
  cout << "temp = temp + temp2: " << temp << endl;

  temp = temp + 23.0f;
  cout << "temp = temp + 23.0f: " << temp << endl;

  temp = temp - 23;
  cout << "temp = temp - 23: " << temp << endl;

  cout << "temp1: " << temp1 << endl;
  cout << "temp2: " << temp2 << endl;
  cout << "temp3: " << temp3 << endl;
  cout << "temp4: " << temp4 << endl;

  int num_int, res;
  res = temp1.toInt(num_int);
  cout << "temp1.toInt(num_int), res: " << res << " data: " << num_int << endl;

  float num_flt;
  res = temp2.toFloat(num_flt);
  cout << "temp2.toFloat(num_flt), res: " << res << " data: " << num_flt << endl;

  res = temp1.toFloat(num_flt);
  cout << "temp1.toFloat(num_flt), res: " << res << " data: " << num_flt << endl;

  res = temp2.toInt(num_int);
  cout << "temp2.toInt(num_int), res: " << res << " data: " << num_int << endl;

  res = temp1.find(567);
  cout << "temp1.find(567), res: " << res << endl;

  char *p;
  p = temp3.subString(1, 2);
  cout << "temp3.subString(1, 2), result: " << p << endl;
  return 0;
}