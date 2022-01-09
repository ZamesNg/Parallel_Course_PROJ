#ifndef _MYSTRING_HPP
#define _MYSTRING_HPP

class MyString
{
private:
  const static int STRING_BUFF_SIZE;
  int len;
  char *_string;

public:
  MyString();
  MyString(int s);
  MyString(float s);
  MyString(const char* s);
  MyString(const MyString &s);
  ~MyString();
};

#endif