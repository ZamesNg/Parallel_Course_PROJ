#include "comm_manager.hpp"

#define MAX_THREADS 64
#define SUBDATANUM 2000000
#define DATANUM (SUBDATANUM * MAX_THREADS)

float *raw_data;

int main(int argc, char **argv)
{
  raw_data = new float[DATANUM];
  CommManager server(CommManager::SERVER, "192.168.3.15", 23107);

  for (int i = 0; i < DATANUM; ++i)
    raw_data[i] = i + 1;

  for (int i = 0; i < 3; i++)
    printf("%f \t %f \t %f \t %f\r\n", raw_data[4 * i], raw_data[4 * i + 1], raw_data[4 * i + 2], raw_data[4 * i + 3]);

  int cmd = 2000;
  server.SendInt(&cmd);
  printf("send %d! \r\n", cmd);

  cmd = server.GetInt();
  printf("get %d! \r\n", cmd);

  if (cmd == 2001)
    server.SendFloatArray(raw_data, sizeof(float) * DATANUM);

  while (true)
  {
  }
  return 0;
}