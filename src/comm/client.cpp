#include "comm_manager.hpp"

#define MAX_THREADS 64
#define SUBDATANUM 2000000
#define DATANUM (SUBDATANUM * MAX_THREADS)

float *raw_data;

int main(int argc, char **argv)
{
  raw_data = new float[DATANUM];
  CommManager client(CommManager::CLIENT, "192.168.3.15", 23107);
  size_t cnt = 0;
  int cmd = 0;
  client.SetRecvDataPtr(raw_data);

  cmd = client.GetInt();
  printf("get %d! \r\n", cmd);

  if (cmd == 2000)
  {
    cmd = 2001;
    client.SendInt(&cmd);
    printf("send %d! \r\n", cmd);
  }

  cnt = client.GetFLoatArray();
  if (cnt)
  {
    printf("get:%d \r\n", cnt);
    for (int i = 0; i < 3; i++)
      printf("%f \t %f \t %f \t %f\r\n", raw_data[4 * i], raw_data[4 * i + 1], raw_data[4 * i + 2], raw_data[4 * i + 3]);
  }

  while (true)
  {
  }
  return 0;
}