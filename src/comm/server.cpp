#include "comm_manager.hpp"

#define MAX_THREADS 64
#define SUBDATANUM 2000000
#define DATANUM (SUBDATANUM * MAX_THREADS)

float *raw_data_send;
float *raw_data_recv;

int main(int argc, char **argv)
{
  raw_data_send = new float[DATANUM];
  raw_data_recv = new float[DATANUM];
  CommManager server(CommManager::SERVER, "192.168.133.21", 30000);
  server.SetRecvDataPtr(raw_data_recv);

  size_t cnt = 0;
  int cmd = 2000;
  server.SendInt(&cmd);
  printf("send %d! \r\n", cmd);

  for (int i = 0; i < DATANUM; ++i)
    raw_data_send[i] = i + 1;

  for (int i = 0; i < 3; i++)
    printf("%f \t %f \t %f \t %f\r\n", raw_data_send[4 * i], raw_data_send[4 * i + 1], raw_data_send[4 * i + 2], raw_data_send[4 * i + 3]);

  cmd = server.GetInt();
  printf("get %d! \r\n", cmd);

  if (cmd == 2000)
  {
    cmd = 2001;
    server.SendInt(&cmd);
    printf("send %d! \r\n", cmd);

    server.SendFloatArray(raw_data_send, sizeof(float) * DATANUM);
    cnt = server.GetFLoatArray();
  }
  if (cnt)
  {
    printf("get:%d \r\n", cnt);
    for (int i = 0; i < 3; i++)
      printf("%f \t %f \t %f \t %f\r\n", raw_data_send[4 * i], raw_data_send[4 * i + 1], raw_data_send[4 * i + 2], raw_data_send[4 * i + 3]);
  }else{
    printf("get nothing...\r\n");
  }

  // server.SendFloatArray(raw_data, sizeof(float) * DATANUM);

  while (true)
  {
  }
  return 0;
}