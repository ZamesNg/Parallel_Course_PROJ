#include <iostream>
#include <omp.h>

#include "comm/comm_manager.hpp"
#include "max/max.hpp"
#include "sum/sum.hpp"
#include "sort/sort.hpp"

using namespace std;

CommManager *comm;

bool SelectItem();

int main(int argc, char **argv)
{
  int main_entered;

  int comm_type;
  char *socket_ip;
  u_short socket_port;

  wcout.unsetf(ios::scientific);
  wcout.precision(9);
  wcout << L"init data and device..\n";
  InitialCuda(0);
  InitData();

  bool status = 1;
  while (status)
    status = SelectItem();

  ReleaseCuda();

  return 0;
}

bool MaxLoop()
{
  bool flag = true;
  int item;
  float max;
  double begin_t, finish_t;
  wcout << L"selecting the item: \n\
                    1. origin Max() with cpu \n\
                    2. Max() speed up with cuda \n\
                    3. Max() speed up with avx \n\
                    4. Max() speed up with avx and omp \n\
                    5. return; \n\
                    input: ";
  wcin >> item;
  for (int i = 0; i < 5; i++)
  {
    switch (item)
    {
    case 1:
      begin_t = omp_get_wtime();
      max = Max(rawFloatData, DATANUM);
      finish_t = omp_get_wtime();
      break;
    case 2:
      begin_t = omp_get_wtime();
      MaxWithCuda(&max, rawFloatData, DATANUM);
      finish_t = omp_get_wtime();
      break;
    case 3:
      begin_t = omp_get_wtime();
      max = MaxSpeedUpAvx(rawFloatData, DATANUM);
      finish_t = omp_get_wtime();
      break;
    case 4:
      begin_t = omp_get_wtime();
      max = MaxSpeedUpAvxOmp(rawFloatData, DATANUM);
      finish_t = omp_get_wtime();
      break;
    default:
      flag = false;
      break;
    }
    wcout << "max number is:" << max << " .\t";
    wcout << "time consumption is " << finish_t - begin_t << "s" << endl;
  }
  return flag;
}

bool SumLoop()
{
  bool flag = true;
  int item;
  float sum;
  double begin_t, finish_t;
  wcout << L"selecting the item: \n\
                    1. origin Sum() with cpu \n\
                    2. Sum() speed up with cuda \n\
                    3. Sum() speed up with avx \n\
                    4. Sum() speed up with avx and omp \n\
                    5. return; \n\
                    input: ";
  wcin >> item;
  for (int i = 0; i < 5; i++)
  {
    switch (item)
    {
    case 1:
      begin_t = omp_get_wtime();
      sum = Sum(rawFloatData, DATANUM);
      finish_t = omp_get_wtime();
      break;
    case 2:
      begin_t = omp_get_wtime();
      SumWithCuda(&sum, rawFloatData, DATANUM);
      finish_t = omp_get_wtime();
      break;
    case 3:
      begin_t = omp_get_wtime();
      sum = SumSpeedUpAvx(rawFloatData, DATANUM);
      finish_t = omp_get_wtime();
      break;
    case 4:
      begin_t = omp_get_wtime();
      sum = SumSpeedUpAvxOmp(rawFloatData, DATANUM);
      finish_t = omp_get_wtime();
      break;
    default:
      flag = false;
      break;
    }
    wcout << "sum is:" << sum << " .\t";
    wcout << "time consumption is " << finish_t - begin_t << "s" << endl;
  }
  return flag;
}

bool SortLoop()
{
  bool flag = true, chk = false;
  int item;
  double begin_t, finish_t;
  wcout << L"selecting the item: \n\
                    1. origin Bitonic Sort() with cpu \n\
                    2. Bitonic Sort() speed up with cuda \n\
                    3. Bitonic Sort() speed up with cuda and 2 computers \n\
                    4. return; \n\
                    input: ";
  wcin >> item;
  switch (item)
  {
  case 1:
    begin_t = omp_get_wtime();
    Sort(rawFloatData, DATANUM);
    finish_t = omp_get_wtime();
    chk = CheckSortResult(rawFloatData, DATANUM, true);
    wcout << "check result is:" << chk << " .\t";
    wcout << "time consumption is " << finish_t - begin_t << "s" << endl;
    break;
  case 2:
    begin_t = omp_get_wtime();
    SortWithCuda(rawFloatData, DATANUM, true);
    finish_t = omp_get_wtime();
    chk = CheckSortResult(rawFloatData, DATANUM, true);
    wcout << "check result is:" << chk << " .\t";
    wcout << "time consumption is " << finish_t - begin_t << "s" << endl;
    break;
  case 3:
    int type;
    wcout << L"selecting communication type: \n\
                    1. server \n\
                    2. client \n\
                    input: ";
    wcin >> type;
    wchar_t *socket_ip;
    wcout << L"\nIP addr: ";

    socket_ip = new wchar_t[16];
    wcin.get();
    wcin.get(socket_ip, 16);
    char ip[16];
    wcstombs(ip, socket_ip, 16);

    uint16_t socket_port;
    wcout << L"\nport: ";
    wcin >> socket_port;

    float *raw_data_recv;
    int cmd;
    raw_data_recv = new float[2 * DATANUM];

    if (type == 1)
    {
      comm = new CommManager(CommManager::SERVER, ip, socket_port);
      comm->SetRecvDataPtr(raw_data_recv);

      // send init cmd;
      cmd = 2000;
      comm->SendInt(&cmd);
      wcout << "send: " << cmd << endl;

      begin_t = omp_get_wtime();
      SortWithCuda(rawFloatData, DATANUM, true);
      finish_t = omp_get_wtime();
      wcout << "sort time consumption is " << finish_t - begin_t << "s" << endl;

      // wait start cmd
      cmd = comm->GetInt();
      wcout << "get: " << cmd << endl;
      if (cmd == 2000)
      {
        cmd = 2001;
        comm->SendInt(&cmd);
        wcout << "send: " << cmd << endl;
        comm->SendFloatArray(rawFloatData, sizeof(float) * DATANUM);

        begin_t = omp_get_wtime();
        comm->GetFLoatArray();
        finish_t = omp_get_wtime();
        wcout << "recv time consumption is " << finish_t - begin_t << "s" << endl;

        begin_t = omp_get_wtime();
        MergeTwoSortedArray(raw_data_recv, DATANUM, rawFloatData, DATANUM);
        finish_t = omp_get_wtime();
        wcout << "merge time consumption is " << finish_t - begin_t << "s" << endl;
        chk = CheckSortResult(raw_data_recv, 2 * DATANUM, true);
        wcout << "check result is:" << chk << " .\t" << endl;
      }
      else
      {
        flag = false;
        break;
      }
      cmd = comm->GetInt();
    }
    else if (type == 2)
    {
      comm = new CommManager(CommManager::CLIENT, ip, socket_port);
      comm->SetRecvDataPtr(raw_data_recv);

      cmd = comm->GetInt();
      wcout << "get: " << cmd << endl;
      if (cmd == 2000)
      {
        begin_t = omp_get_wtime();
        SortWithCuda(rawFloatData, DATANUM, true);
        finish_t = omp_get_wtime();
        wcout << "sort time consumption is " << finish_t - begin_t << "s" << endl;

        comm->SendInt(&cmd);
        wcout << "send: " << cmd << endl;
      }
      else
      {
        flag = false;
        break;
      }

      cmd = comm->GetInt();
      wcout << "get: " << cmd << endl;
      if (cmd == 2001)
      {
        comm->SendFloatArray(rawFloatData, sizeof(float) * DATANUM);

        begin_t = omp_get_wtime();
        comm->GetFLoatArray();
        finish_t = omp_get_wtime();
        wcout << "recv time consumption is " << finish_t - begin_t << "s" << endl;

        begin_t = omp_get_wtime();
        MergeTwoSortedArray(raw_data_recv, DATANUM, rawFloatData, DATANUM);
        finish_t = omp_get_wtime();
        wcout << "merge time consumption is " << finish_t - begin_t << "s" << endl;
        chk = CheckSortResult(raw_data_recv, 2 * DATANUM, true);
        wcout << "check result is:" << chk << " .\t" << endl;
      }
      else
      {
        flag = false;
        break;
      }
      cmd = 2002;
      comm->SendInt(&cmd);
    }

    delete[] raw_data_recv;
    delete comm;
    raw_data_recv = nullptr;
    comm = nullptr;

    break;

  default:
    flag = false;
    break;
  }
  return flag;
}

bool SelectItem()
{
  int item;
  bool flag;
  wcout << L"selecting the item: \n\
                    1. Max \n\
                    2. Sum \n\
                    3. Sort \n\
                    4. quit; \n\
                    input: ";
  wcin >> item;
  switch (item)
  {
  case 1:
    flag = MaxLoop();
    break;
  case 2:
    flag = SumLoop();
    break;
  case 3:
    flag = SortLoop();
    break;
  default:
    flag = false;
    break;
  }
  return flag;
}