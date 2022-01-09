#include "mystring.hpp"
#include <math.h>
#include <stdio.h>

// Reverses a string 'str' of length 'len'
void reverse(char *str, int len)
{
  int i = 0, j = len - 1, temp;
  while (i < j)
  {
    temp = str[i];
    str[i] = str[j];
    str[j] = temp;
    i++;
    j--;
  }
}

// Converts a given integer x to string str[].
// d is the number of digits required in the output.
// If d is more than the number of digits in x,
// then 0s are added at the beginning.
int IntToStr(int x, char str[], int d = 0)
{
  int i = 0;
  if (x < 0)
    str[i++] = '-';

  while (x)
  {
    str[i++] = (x % 10) + '0';
    x = x / 10;
  }

  // If number of digits required is more, then
  // add 0s at the beginning
  while (i < d)
    str[i++] = '0';

  reverse(str, i);
  str[i] = '\0';
  return i;
}

// Converts a floating-point/double number to a string.
int FloatToStr(float n, char *res, int afterpoint = 0)
{
  // Extract integer part
  int ipart = (int)n;

  // Extract floating part
  float fpart = n - (float)ipart;

  // convert integer part to string
  int i = IntToStr(ipart, res, 0);

  // check for display option after point
  if (afterpoint != 0)
  {
    res[i] = '.'; // add dot

    // Get the value of fraction part upto given no.
    // of points after dot. The third parameter
    // is needed to handle cases like 233.007
    fpart = fpart * pow(10, afterpoint);

    IntToStr((int)fpart, res + i + 1, afterpoint);
  }
  return i + afterpoint + 1;
}

const int MyString::STRING_BUFF_SIZE = 1024;

MyString::MyString()
{
  len = 0;
  _string = nullptr;
}

MyString::MyString(int s)
{
  _string = new char[STRING_BUFF_SIZE];
  len = IntToStr(s, _string, 0);
}

MyString::MyString(float s)
{
  _string = new char[STRING_BUFF_SIZE];
  len = FloatToStr(s, _string, 10);
}

MyString::MyString(float s)
{
  _string = new char[STRING_BUFF_SIZE];
  len = FloatToStr(s, _string, 10);
}

MyString::MyString(const char *s)
{
  len = 0;
  _string = new char[STRING_BUFF_SIZE];
  while (s[len])
  {
    _string[len] = s[len];
    len++;
  }
}

MyString::MyString(const MyString &s)
{
  _string = new char[STRING_BUFF_SIZE];
  len = s.len;
  for (int i = 0; i < len; i++)
  {
    _string[i] = s._string[i];
  }
}

MyString::~MyString()
{
  delete[] _string;
}
