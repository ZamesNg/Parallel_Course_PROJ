#ifndef _COMM_MANAGER_HPP
#define _COMM_MANAGER_HPP

#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <iostream>
#include <string.h>

#define COMMAND_FLOAT ((int)1812)
#define COMMAND_INT ((int)1824)
#define COMMAND_FLARRAY ((int)2012)
#define COMMAND_REQEND ((int)4396)
#define COMMAND_RETEND ((int)4397)
#define COMMAND_ACKEND ((int)4398)

class CommManager
{
public:
  enum CommType
  {
    SERVER,
    CLIENT
  };

  CommManager(CommType type, const char *ip, uint16_t port);
  ~CommManager();

  void SetRecvDataPtr(void *data) { recv_data_ptr = (char *)data; };
  size_t GetFLoatArray();
  float GetFLoat();
  int GetInt();

  void SendInt(const int *data);
  void SendFloat(const float *data);
  void SendFloatArray(const float *data, size_t len);

private:
  /* data */
  static const size_t RECV_BUFFER_SIZE = 1024;
  CommType comm_type;

  int socket_fd;
  int remote_socket_fd;
  struct sockaddr_in addr;
  struct sockaddr_in remote_addr;

  sem_t init_finish_semaphore;
  sem_t send_start_semaphore;
  sem_t send_finish_semaphore;
  sem_t recv_finish_semaphore;

  int recv_data_type;
  size_t recv_data_len;
  // small buffer to recv data like command or len
  char *recv_data_buffer;
  // big buffer out size the class to recv array
  char *recv_data_ptr;

  int send_data_type;
  size_t send_data_len;
  char *send_data_ptr;

  pthread_t recv_id;
  pthread_t send_id;

  static void *RecvThread(void *comm_manager);
  static void *SendThread(void *comm_manager);

  void InitServer();
  void InitClient();
};

#endif