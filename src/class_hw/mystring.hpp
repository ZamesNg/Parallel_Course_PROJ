#ifndef _MYSTRING_HPP
#define _MYSTRING_HPP

#include <iostream>

class MyString {
 private:
  const static int STRING_BUFF_SIZE;
  int len;
  char *_string;

 public:
  MyString();
  MyString(int s);
  MyString(float s);
  MyString(const char *s);
  MyString(const MyString &s);
  ~MyString();

  MyString operator+(const MyString &s);
  MyString operator-(const MyString &s);
  
  friend std::ostream &operator<<(std::ostream &os, const MyString &s) {
    os << s._string;
  };
};

#endif