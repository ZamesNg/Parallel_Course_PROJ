#include "comm_manager.hpp"

using namespace std;
CommManager::CommManager(CommType type, const char *ip, uint16_t port) : comm_type(type)
{
  send_data_ptr = nullptr;
  recv_data_buffer = new char[RECV_BUFFER_SIZE];
  recv_data_ptr = nullptr;
  recv_data_len = 0;
  recv_data_type = -1;

  socket_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (socket_fd == -1)
    cout << "create socket fail: " << strerror(errno) << endl;

  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  addr.sin_addr.s_addr = inet_addr(ip);

  sem_init(&send_start_semaphore, 0, 0);
  sem_init(&send_finish_semaphore, 0, 0);
  sem_init(&recv_finish_semaphore, 0, 0);
  sem_init(&init_finish_semaphore, 0, 0);

  switch (comm_type)
  {
  case CLIENT:
    InitClient();
    break;
  case SERVER:
    InitServer();
    break;
  }

  pthread_create(&recv_id, NULL, RecvThread, (void *)this);
  pthread_create(&send_id, NULL, SendThread, (void *)this);
}

CommManager::~CommManager()
{
  delete recv_data_buffer;
}

void CommManager::InitServer()
{

  if (bind(socket_fd, (struct sockaddr *)&addr, sizeof(addr)) == -1)
    cout << "server bind " << inet_ntoa(addr.sin_addr) << ":" << ntohs(addr.sin_port) << " fail: " << strerror(errno) << endl;
  else
    cout << "server bind " << inet_ntoa(addr.sin_addr) << ":" << ntohs(addr.sin_port) << " success!" << endl;

  listen(socket_fd, 30);
  cout << "listening..." << endl;

  int c = sizeof(struct sockaddr_in);
  remote_socket_fd = accept(socket_fd, (struct sockaddr *)&remote_addr, (socklen_t *)&c);

  cout << "accept..." << endl;

  int bytes_cnt, data_tmp = 123;
  bytes_cnt = send(remote_socket_fd, &data_tmp, sizeof(int), 0);
  bytes_cnt = recv(remote_socket_fd, &data_tmp, sizeof(int), 0);
  bytes_cnt = send(remote_socket_fd, &data_tmp, sizeof(int), 0);

  sem_post(&send_finish_semaphore);
}

void CommManager::InitClient()
{
  if (connect(socket_fd, (struct sockaddr *)&addr, sizeof(addr)) == -1)
  {
    cout << "client connect " << inet_ntoa(addr.sin_addr) << ":" << ntohs(addr.sin_port) << " fail: " << strerror(errno) << endl;
    return;
  }
  else
  {
    cout << "client connect " << inet_ntoa(addr.sin_addr) << ":" << ntohs(addr.sin_port) << " success!" << endl;
  }

  int bytes_cnt, data_tmp = 123;
  bytes_cnt = recv(socket_fd, &data_tmp, sizeof(int), 0);
  bytes_cnt = send(socket_fd, &data_tmp, sizeof(int), 0);
  bytes_cnt = recv(socket_fd, &data_tmp, sizeof(int), 0);

  sem_post(&send_finish_semaphore);
}

void *CommManager::RecvThread(void *comm_manager)
{
  CommManager *ptr = (CommManager *)comm_manager;
  int recv_bytes_cnt;

  sem_wait(&ptr->init_finish_semaphore);

  int socket_fd;
  switch (ptr->comm_type)
  {
  case CLIENT:
    /* code */
    socket_fd = ptr->socket_fd;
    break;
  case SERVER:
    /* code */
    socket_fd = ptr->remote_socket_fd;
    break;
  }

  while (true)
  {
    // blocking to recv data type
    recv_bytes_cnt = recv(socket_fd, &ptr->recv_data_type, sizeof(int), 0);
    if (recv_bytes_cnt != sizeof(int))
      cout << "recv data type fail: " << strerror(errno) << endl;
    // recv data len
    recv_bytes_cnt = recv(socket_fd, &ptr->recv_data_len, sizeof(size_t), 0);
    if (recv_bytes_cnt != sizeof(size_t))
      cout << "recv data len fail: " << strerror(errno) << endl;

    char *data_ptr = nullptr;
    if (ptr->recv_data_len <= RECV_BUFFER_SIZE)
      data_ptr = ptr->recv_data_buffer;
    else
      data_ptr = ptr->recv_data_ptr;

    // try best to recv data
    int cnt = 0;
    if (data_ptr)
    {
      while (cnt < ptr->recv_data_len)
      {
        recv_bytes_cnt = recv(socket_fd, data_ptr + cnt, ptr->recv_data_len - cnt, 0);
        cnt += recv_bytes_cnt;
      }
    }
    else
    {
      cout << "recv data ptr not set!" << endl;
      continue;
    }
    // todo: check if will get too much data
    sem_post(&ptr->recv_finish_semaphore);
  }
}

void *CommManager::SendThread(void *comm_manager)
{
  CommManager *ptr = (CommManager *)comm_manager;
  int send_bytes_cnt;
  int socket_fd;
  switch (ptr->comm_type)
  {
  case CLIENT:
    /* code */
    // ptr->InitClient();
    socket_fd = ptr->socket_fd;
    break;
  case SERVER:
    /* code */
    // ptr->InitServer();
    socket_fd = ptr->remote_socket_fd;
    break;
  }

  sem_post(&ptr->init_finish_semaphore);

  while (true)
  {
    // blocking to wait cmd
    sem_wait(&ptr->send_start_semaphore);

    send_bytes_cnt = send(socket_fd, &ptr->send_data_type, sizeof(int), 0);
    if (send_bytes_cnt != sizeof(int))
      cout << "send data type fail: " << strerror(errno) << endl;

    send_bytes_cnt = send(socket_fd, &ptr->send_data_len, sizeof(size_t), 0);
    if (send_bytes_cnt != sizeof(size_t))
      cout << "send data len fail: " << strerror(errno) << endl;

    size_t cnt = 0;
    char *data_ptr = ptr->send_data_ptr;
    if (ptr->send_data_ptr)
    {
      while (cnt < ptr->send_data_len)
      {
        send_bytes_cnt = send(socket_fd, data_ptr + cnt, ptr->send_data_len - cnt, 0);
        cnt += send_bytes_cnt;
      }
    }
    sem_post(&ptr->send_finish_semaphore);
  }
}

// returen the recv array len but not ptr
// the recv data addr should be set using
// CommManager::SetRecvDataPtr()

size_t CommManager::GetFLoatArray()
{
  // blocking to wait recv finish
  sem_wait(&recv_finish_semaphore);
  return recv_data_len / sizeof(float);
}

int CommManager::GetInt()
{
  // blocking to wait recv finish
  sem_wait(&recv_finish_semaphore);
  if (recv_data_type == COMMAND_INT)
    return *(int *)recv_data_buffer;
  else
  {
    cout << "error type" << endl;
    return -1;
  }
}

float CommManager::GetFLoat()
{
  // blocking to wait recv finish
  sem_wait(&recv_finish_semaphore);
  if (recv_data_type == COMMAND_FLOAT)
    return *(float *)recv_data_buffer;
  else
  {
    cout << "error type" << endl;
    return -1.0f;
  }
}

void CommManager::SendInt(const int *data)
{
  // blocking to wait send finish
  sem_wait(&send_finish_semaphore);

  send_data_type = COMMAND_INT;
  send_data_ptr = (char *)data;
  send_data_len = sizeof(int);

  // start a new sending
  sem_post(&send_start_semaphore);
}

void CommManager::SendFloat(const float *data)
{
  // blocking to wait send finish
  sem_wait(&send_finish_semaphore);

  send_data_type = COMMAND_FLOAT;
  send_data_ptr = (char *)data;
  send_data_len = sizeof(float);

  // start a new sending
  sem_post(&send_start_semaphore);
}
void CommManager::SendFloatArray(const float *data, size_t len)
{
  // blocking to wait send finish
  sem_wait(&send_finish_semaphore);

  send_data_type = COMMAND_FLARRAY;
  send_data_ptr = (char *)data;
  send_data_len = len;

  // start a new sending
  sem_post(&send_start_semaphore);
}