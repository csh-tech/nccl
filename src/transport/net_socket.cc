/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "nccl.h"
#include "core.h"
#include "socket.h"
#include "net.h"

#include <assert.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <poll.h>
#include <limits.h>

/* Init functions */
static char ncclNetIfNames[MAX_IF_NAME_SIZE*MAX_IFS];  // 存储interface names
static union socketAddress ncclNetIfAddrs[MAX_IFS];    // 存储interface address
static int ncclNetIfs = -1;  // 记录interface的总数
pthread_mutex_t ncclSocketLock = PTHREAD_MUTEX_INITIALIZER;  // 初始化mutex

// 初始化socket，调用findInterfaces()函数搜索本地interfaces，并将interfaec names保存在数组总
ncclResult_t ncclSocketInit(ncclDebugLogger_t logFunction) { 
  if (ncclNetIfs == -1) {
    pthread_mutex_lock(&ncclSocketLock);  // 锁定mutex
    if (ncclNetIfs == -1) {
      // 寻找符合条件的interface，由ncclNetIfNames和ncclNetIfAddrs记录相关信息，ncclNetIfs返回interface数量，
      ncclNetIfs = findInterfaces(ncclNetIfNames, ncclNetIfAddrs, MAX_IF_NAME_SIZE, MAX_IFS);  
      if (ncclNetIfs <= 0) {
        WARN("NET/Socket : no interface found");
        return ncclInternalError;
      } else {
        char line[1024];
        char addrline[1024];
        line[0] = '\0';  // 以'\0'开头
        for (int i=0; i<ncclNetIfs; i++) {
          // 将寻找到的interface的name写入line数组中
          snprintf(line+strlen(line), 1023-strlen(line), " [%d]%s:%s", i, ncclNetIfNames+i*MAX_IF_NAME_SIZE,
              socketToString(&ncclNetIfAddrs[i].sa, addrline));
        }
        line[1023] = '\0';  // 以'\0'结尾
        INFO(NCCL_INIT|NCCL_NET,"NET/Socket : Using%s", line);
      }
    }
    pthread_mutex_unlock(&ncclSocketLock);  // 解锁mutex
  }
  return ncclSuccess;
}

//这个函数是干什么的？
ncclResult_t ncclSocketPtrSupport(int dev, int* supportedTypes) {
  *supportedTypes = NCCL_PTR_HOST;
  return ncclSuccess;
}

// *ndev = ncclNetIfs (ncclNetIfs记录interface总数)
ncclResult_t ncclSocketDevices(int* ndev) { // *ndev将保存interface的总数
  *ndev = ncclNetIfs;
  return ncclSuccess;
}

// 返回序号为dev的interface的绝对路径path（interface names全部存储在ncclNeiIfNames中，interface name的序号为dev）
ncclResult_t ncclSocketPciPath(int dev, char** path) {
  char devicepath[PATH_MAX];
  snprintf(devicepath, PATH_MAX, "/sys/class/net/%s/device", ncclNetIfNames+dev*MAX_IF_NAME_SIZE);
  *path = realpath(devicepath, NULL);
  if (*path == NULL) {
    INFO(NCCL_NET|NCCL_INIT, "Could not find real path of %s", devicepath);
    return ncclSystemError;
  }
  return ncclSuccess;
}

// 从ncclNetIfAddr数组中得到序号为dev的interface的address信息，包括family、port、internet address等，保存在addr中
static ncclResult_t GetSocketAddr(int dev, union socketAddress* addr) {
  if (dev >= ncclNetIfs) return ncclInternalError;
  memcpy(addr, ncclNetIfAddrs+dev, sizeof(*addr));
  return ncclSuccess;
}

/* Communication functions */

// socket的句柄，存储要连接的Addr
struct ncclSocketHandle { 
  union socketAddress connectAddr;  // local address，或者remote address，存在两种模式，待续。。。
};

struct ncclSocketRequest {  // socket request
  int op;
  void* data;
  int size;
  int fd;
  int offset;
  int used;
};

// socket requests
struct ncclSocketReqs {
  struct ncclSocketRequest* requests;
};

struct ncclSocketComm {
  int fd;
  struct ncclSocketReqs reqs;
};

// 为socketComm申请空间，并将fd置为-1
ncclResult_t ncclSocketNewComm(struct ncclSocketComm** comm) {
  NCCLCHECK(ncclCalloc(comm, 1));  // 利用ncclCalloc()为comm申请空间
  (*comm)->fd = -1;
  return ncclSuccess;
}

// create handle，str中存储的是ip_port_pair，将信息都赋值给handle->connectAddr (localAddr？)
ncclResult_t ncclSocketCreateHandle(void* opaqueHandle, const char* str) {  
  struct ncclSocketHandle* handle = (struct ncclSocketHandle*) opaqueHandle;
  NCCLCHECK(GetSocketAddrFromString(&(handle->connectAddr), str));
  return ncclSuccess;
}

// 指定序号为dev的interface进行listen
ncclResult_t ncclSocketListen(int dev, void* opaqueHandle, void** listenComm) {  
  struct ncclSocketHandle* handle = (struct ncclSocketHandle*) opaqueHandle;
  static_assert(sizeof(struct ncclSocketHandle) < NCCL_NET_HANDLE_MAXSIZE, "ncclSocketHandle size too large");
  // if dev >= 0, listen based on dev ？？？
  if (dev >= 0) {
    // 将序号为dev的interface相关信息存储到handle的connectAddr中（从ncclNetIfAddrs中读取，序号为dev）
    NCCLCHECK(GetSocketAddr(dev, &(handle->connectAddr)));    // 这种情况下，opaqueHandle会被重新赋值，或者被覆盖
  } else if (dev == findSubnetIf) {  // 当dev=-1时，二者相等，启用findSubnet模式
    // 这种模式下，handle中存储的是remote address，需要调用findInterfaceMatchSubnet()
    // 函数寻找其在相同子网下的local address
    // handle stores a remote address
    // need to find a local addr that is in the same network as the remote addr
    union socketAddress localAddr;
    char ifName[MAX_IF_NAME_SIZE];
    if (findInterfaceMatchSubnet(ifName, &localAddr, handle->connectAddr, MAX_IF_NAME_SIZE, 1) <= 0) {
      WARN("NET/Socket : No usable listening interface found");
      return ncclSystemError;
    }
    // pass the local address back
    // 将寻找到的local address赋值给handle->connectAddress
    memcpy(&handle->connectAddr, &localAddr, sizeof(handle->connectAddr));
  } // Otherwise, handle stores a local address
  struct ncclSocketComm* comm;
  NCCLCHECK(ncclSocketNewComm(&comm));
  NCCLCHECK(createListenSocket(&comm->fd, &handle->connectAddr));
  *listenComm = comm;
  return ncclSuccess;
}

ncclResult_t ncclSocketConnect(int dev, void* opaqueHandle, void** sendComm) {  // opaqueHandle是否是remote handle?
  struct ncclSocketComm* comm;
  NCCLCHECK(ncclSocketNewComm(&comm));
  struct ncclSocketHandle* handle = (struct ncclSocketHandle*) opaqueHandle;
  NCCLCHECK(connectAddress(&comm->fd, &handle->connectAddr));  // handle->connectAddr中存储的是remoteAddr，函数返回comm->fd
  *sendComm = comm;
  return ncclSuccess;
}

ncclResult_t ncclSocketAccept(void* listenComm, void** recvComm) {
  struct ncclSocketComm* lComm = (struct ncclSocketComm*)listenComm;
  struct ncclSocketComm* rComm;
  NCCLCHECK(ncclSocketNewComm(&rComm));
  struct sockaddr_in sockaddr;
  socklen_t socklen = sizeof(struct sockaddr_in);
  /*
  int accept(int sockfd, struct sockaddr *addr, socklen_t *addrlen)：（服务端）
  TCP服务器端依次调用socket()、bind()、listen()之后，就会监听指定的socket地址了。
  TCP客户端依次调用socket()、connect()之后就想TCP服务器发送了一个连接请求。TCP服
  务器监听到这个请求之后，就会调用accept()函数取接收请求，这样连接就建立好了。之后
  就可以开始网络I/O操作了，即类同于普通文件的读写I/O操作。
  accept的第一个参数为服务器的socket描述字，是服务器开始调用socket()函数生成的，
  称为监听socket描述字；而accept函数返回的是已连接的socket描述字。一个服务器通常通
  常仅仅只创建一个监听socket描述字，它在该服务器的生命周期内一直存在。内核为每个由服
  务器进程接受的客户连接创建了一个已连接socket描述字，当服务器完成了对某个客户的服务，
  相应的已连接socket描述字就被关闭。
  */
  SYSCHECKVAL(accept(lComm->fd, (struct sockaddr*)&sockaddr, &socklen), "accept", rComm->fd);
  // 上一行代码执行的过程中rComm->fd已经被重新赋值为内核重新创建的已连接的套接字，sockaddr将存储客户端协议地址
  *recvComm = rComm;
  return ncclSuccess;
}

#define MAX_REQUESTS 128

ncclResult_t ncclSocketGetRequest(struct ncclSocketReqs* reqs, int op, void* data, int size, int fd, struct ncclSocketRequest** req) {
  if (reqs->requests == NULL) {
    NCCLCHECK(ncclCalloc(&reqs->requests, MAX_REQUESTS));  // 为requests申请空间
  }
  for (int i=0; i<MAX_REQUESTS; i++) {
    struct ncclSocketRequest* r = reqs->requests+i;
    if (r->used == 0) {  // 寻找一个空的位置保存req
      r->op = op;
      r->data = data;
      r->size = size;
      r->fd = fd;
      r->offset = -1;
      r->used = 1;
      *req = r;
      return ncclSuccess;
    }
  }
  WARN("Socket : unable to allocate requests");
  return ncclInternalError;
}

ncclResult_t ncclSocketTest(void* request, int* done, int* size) {
  *done = 0;
  struct ncclSocketRequest *r = (struct ncclSocketRequest*)request;
  if (r == NULL) {
    WARN("NET/Socket : test called with NULL request");
    return ncclInternalError;
  }
  if (r->offset == -1) { /* try to send/recv size */
    int data = r->size;
    int offset = 0;
    NCCLCHECK(socketProgress(r->op, r->fd, &data, sizeof(int), &offset));

    if (offset == 0) return ncclSuccess; /* Not ready -- retry later */

    // Not sure we could ever receive less than 4 bytes, but just in case ...
    if (offset < sizeof(int)) NCCLCHECK(socketWait(r->op, r->fd, &data, sizeof(int), &offset));

    // Check size is less or equal to the size provided by the user
    if (r->op == NCCL_SOCKET_RECV && data > r->size) {
      WARN("NET/Socket : message truncated : receiving %d bytes instead of %d", data, r->size);
      return ncclInternalError;
    }
    r->size = data;
    r->offset = 0;
  }
  if (r->offset < r->size) {
    NCCLCHECK(socketProgress(r->op, r->fd, r->data, r->size, &r->offset));
  }
  if (r->offset == r->size) {
    if (size) *size = r->size;
    *done = 1;
    r->used = 0;
  }
  return ncclSuccess;
}

ncclResult_t ncclSocketRegMr(void* comm, void* data, int size, int type, void** mhandle) {
  return (type != NCCL_PTR_HOST) ? ncclInternalError : ncclSuccess;
}
ncclResult_t ncclSocketDeregMr(void* comm, void* mhandle) { return ncclSuccess; }

ncclResult_t ncclSocketIsend(void* sendComm, void* data, int size, void* mhandle, void** request) {
  struct ncclSocketComm* comm = (struct ncclSocketComm*)sendComm;
  NCCLCHECK(ncclSocketGetRequest(&comm->reqs, NCCL_SOCKET_SEND, data, size, comm->fd, (struct ncclSocketRequest**)request));
  return ncclSuccess;
}

ncclResult_t ncclSocketIrecv(void* recvComm, void* data, int size, void* mhandle, void** request) {
  struct ncclSocketComm* comm = (struct ncclSocketComm*)recvComm;
  NCCLCHECK(ncclSocketGetRequest(&comm->reqs, NCCL_SOCKET_RECV, data, size, comm->fd, (struct ncclSocketRequest**)request));
  return ncclSuccess;
}

ncclResult_t ncclSocketFlush(void* recvComm, void* data, int size, void* mhandle) {
  // We don't support CUDA pointers, so we don't need a flush operation
  return ncclInternalError;
}

ncclResult_t ncclSocketClose(void* opaqueComm) {
  struct ncclSocketComm* comm = (struct ncclSocketComm*)opaqueComm;
  if (comm) {
    free(comm->reqs.requests);
    close(comm->fd);
    free(comm);
  }
  return ncclSuccess;
}

ncclNet_t ncclNetSocket = {
  "Socket",
  ncclSocketInit,
  ncclSocketDevices,
  ncclSocketPciPath,
  ncclSocketPtrSupport,
  ncclSocketListen,
  ncclSocketConnect,
  ncclSocketAccept,
  ncclSocketRegMr,
  ncclSocketDeregMr,
  ncclSocketIsend,
  ncclSocketIrecv,
  ncclSocketFlush,
  ncclSocketTest,
  ncclSocketClose,
  ncclSocketClose,
  ncclSocketClose
};
