/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_SOCKET_H_
#define NCCL_SOCKET_H_

#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/tcp.h>
#include <unistd.h>
#include <netdb.h>
#include <ifaddrs.h>
#include <net/if.h>
#include "utils.h"

#define MAX_IFS 16
#define MAX_IF_NAME_SIZE 16
#define SLEEP_INT            1000 // connection retry sleep interval in usec
#define RETRY_REFUSED_TIMES   2e4 // connection refused retry times before reporting a timeout (20 sec)
#define RETRY_TIMEDOUT_TIMES    3 // connection timed out retry times (each one can take 20s)

/* Common socket address storage structure for IPv4/IPv6 */
// sa sin sin6三者共享内存，所以同一时刻三者中只有一个值是有效的
// 内存大小取三者内存的最大值，即max(sizeof(sa), sizeof(sin), sizeof(sin6))

/* Format a string representation of a (struct sockaddr *) socket address using getnameinfo()
 *
 * Output: "IPv4/IPv6 address<port>"
 */
static inline const char *socketToString(struct sockaddr *saddr, char *buf) {
  if (buf == NULL || saddr == NULL) return NULL;
  if (saddr->sa_family != AF_INET && saddr->sa_family != AF_INET6) { buf[0]='\0'; return buf; }
  char host[NI_MAXHOST], service[NI_MAXSERV];
  (void) getnameinfo(saddr, sizeof(union socketAddress), host, NI_MAXHOST, service, NI_MAXSERV, NI_NUMERICHOST|NI_NUMERICSERV);
  sprintf(buf, "%s<%s>", host, service);
  return buf;
}

// 传入socket address传出对应的port    被createListenSocket()调用
static inline short socketToPort(struct sockaddr *saddr) {
  return ntohs(saddr->sa_family == AF_INET ? ((struct sockaddr_in*)saddr)->sin_port : ((struct sockaddr_in6*)saddr)->sin6_port);
}  // network to host （网络序 --> 主机序） 

// 用户利用环境变量指定是使用ipv4还是ipv6，被findeInterFaces()调用
/* Allow the user to force the IPv4/IPv6 interface selection */
static inline int envSocketFamily(void) {
  int family = -1; // Family selection is not forced, will use first one found
  char* env = getenv("NCCL_SOCKET_FAMILY");
  if (env == NULL)
    return family;

  if (strcmp(env, "AF_INET") == 0)
    family = AF_INET;  // IPv4
  else if (strcmp(env, "AF_INET6") == 0)
    family = AF_INET6; // IPv6 
  return family;
}

// ifName 中存储网卡名
// ifAddrs 中存储 ip 地址
//搜索本地所有的interface  被findInterfaces()调用
static int findInterfaces(const char* prefixList, char* names, union socketAddress *addrs, int sock_family, int maxIfNameSize, int maxIfs) {
#ifdef ENABLE_TRACE
  char line[1024];
#endif
  struct netIf userIfs[MAX_IFS];
  bool searchNot = prefixList && prefixList[0] == '^';
  int nUserIfs = parseStringList(prefixList, userIfs, MAX_IFS);

  int found = 0;
  struct ifaddrs *interfaces, *interface;
  getifaddrs(&interfaces);
  for (interface = interfaces; interface && found < maxIfs; interface = interface->ifa_next) {
    if (interface->ifa_addr == NULL) continue;

    /* We only support IPv4 & IPv6 */
    int family = interface->ifa_addr->sa_family;
    if (family != AF_INET && family != AF_INET6)
      continue;

    TRACE(NCCL_INIT|NCCL_NET,"Found interface %s:%s", interface->ifa_name, socketToString(interface->ifa_addr, line));

    /* Allow the caller to force the socket family type */
    // 当用户指定socket family时，sock_family != -1
    if (sock_family != -1 && family != sock_family)
      continue;

    /* We also need to skip IPv6 loopback interfaces */
    if (family == AF_INET6) {
      struct sockaddr_in6* sa = (struct sockaddr_in6*)(interface->ifa_addr);
      // 如果地址是环回IPv6地址，则IN6_IS_ADDR_LOOPBACK返回true，否则返回false。  netinet/in.h
      if (IN6_IS_ADDR_LOOPBACK(&sa->sin6_addr)) continue;  
    }

    // check against user specified interfaces
    // 如果不是用户至指定的interface，扔掉
    if (!(matchIfList(interface->ifa_name, -1, userIfs, nUserIfs) ^ searchNot)) {
      continue;
    }

    // Check that this interface has not already been saved
    // getifaddrs() normal order appears to be; IPv4, IPv6 Global, IPv6 Link
    // 判断是否和已经找到的网卡重复
    bool duplicate = false;
    for (int i = 0; i < found; i++) {
      if (strcmp(interface->ifa_name, names+i*maxIfNameSize) == 0) { duplicate = true; break; }
    }
    // 不重复时将结果添加到可用网卡组中
    if (!duplicate) {
      // Store the interface name
      strncpy(names+found*maxIfNameSize, interface->ifa_name, maxIfNameSize);
      // Store the IP address
      int salen = (family == AF_INET) ? sizeof(sockaddr_in) : sizeof(sockaddr_in6);
      memcpy(addrs+found, interface->ifa_addr, salen);
      found++;
    }
  }

  freeifaddrs(interfaces);  // 释放空间
  return found;
}

// 判断local interface和remote interface是否在同一子网下  被findInterfaceMatchSubnet()调用
static bool matchSubnet(struct ifaddrs local_if, union socketAddress remote) {
  /* Check family first */
  int family = local_if.ifa_addr->sa_family;
  if (family != remote.sa.sa_family) {
    return false;
  }

  // 子网掩码例子：
  // 当子网掩码为255.255.255.252时，子网是如何划分的？
  // 将该子网掩码转换成二进制为30个1和2个0，
  // 表示每个子网中只有4个IP地址（2的2次方），192.168.1.0-255的地址段共可划分64个子网，
  // 第一个子网的地址范围是192.168.1.0-192.168.1.3，
  // 第二个子网的地址范围是192.168.1.4-192.168.1.7，
  // 依次类推。其中每个子网第一个和最后一个IP地址不可用(规定每个子网的第一个IP地址为网段地址，
  // 最后一个IP地址为广播地址，都不可用)，可用的只有2个IP地址。也就是说：
  // 如果子网掩码设置为255.255.255.252，那么该子网只能容纳两台电脑，
  // 而且这两台电脑的IP必须在一个子网内才能正常联网，例如一台电脑的IP设为192.168.1.10，
  // 另外一台电脑的IP必须设置为192.168.1.9。

  if (family == AF_INET) { 
    struct sockaddr_in* local_addr = (struct sockaddr_in*)(local_if.ifa_addr);
    struct sockaddr_in* mask = (struct sockaddr_in*)(local_if.ifa_netmask);
    struct sockaddr_in& remote_addr = remote.sin;
    struct in_addr local_subnet, remote_subnet;
    /**
     * 子网掩码：255.255.255.252    11111111.11111111.11111111.11111100
     * 网段： 192.168.1.0
     * 4个ip为一个子网，如192.168.1.0，192.168.1.1，192.168.1.2，192.168.1.3为同一子网，
     * 但是192.168.1.0和192.168.1.3不可用
     * 可以利用子网掩码判断两个ip是否属于网之
     * 例如：192.168.1.1和192.168.1.2在同一子网中，所以与子网掩码位与的结果相同
     * 11000000.10101000.00000001.000000 01 & 11111111.11111111.11111111.111111 00 --> 11000000.10101000.00000001.000000 00
     * 11000000.10101000.00000001.000000 10 & 11111111.11111111.11111111.111111 00 --> 11000000.10101000.00000001.000000 00
     * 结果进行异或：
     * 11000000.10101000.00000001.000000 00 ^ 11000000.10101000.00000001.000000 00 --> 00000000.00000000.00000000.000000000
     * 全0 --> 同一子网
     * 而192.168.1.1和192.168.1.5不在同一子网中，所以与子网掩码位与的结果不相同
     * 11000000.10101000.00000001.000000 01 & 11111111.11111111.11111111.111111 00 --> 11000000.10101000.00000001.000000 00
     * 11000000.10101000.00000001.000001 01 & 11111111.11111111.11111111.111111 00 --> 11000000.10101000.00000001.000001 00
     * 结果进行异或：
     * 11000000.10101000.00000001.000000 00 ^ 11000000.10101000.00000001.000001 00 --> 00000000.00000000.00000001.000000000
     * 非全0 --> 非同一子网                                                                                                      |
     * */
    local_subnet.s_addr = local_addr->sin_addr.s_addr & mask->sin_addr.s_addr;  // 与子网掩码进行 位与 运算，判断是否属于该子网
    remote_subnet.s_addr = remote_addr.sin_addr.s_addr & mask->sin_addr.s_addr;  // 位运算符 与
    return (local_subnet.s_addr ^ remote_subnet.s_addr) ? false : true;  // ^ 位运算符 异或
  } else if (family == AF_INET6) {
    struct sockaddr_in6* local_addr = (struct sockaddr_in6*)(local_if.ifa_addr);
    struct sockaddr_in6* mask = (struct sockaddr_in6*)(local_if.ifa_netmask);
    struct sockaddr_in6& remote_addr = remote.sin6;
    struct in6_addr& local_in6 = local_addr->sin6_addr;
    struct in6_addr& mask_in6 = mask->sin6_addr;
    struct in6_addr& remote_in6 = remote_addr.sin6_addr;
    bool same = true;
    int len = 16;  //IPv6 address is 16 unsigned char
    for (int c = 0; c < len; c++) {  //Network byte order is big-endian
      char c1 = local_in6.s6_addr[c] & mask_in6.s6_addr[c];
      char c2 = remote_in6.s6_addr[c] & mask_in6.s6_addr[c];
      if (c1 ^ c2) {
        same = false;
        break;
      }
    }
    // At last, we need to compare scope id
    // Two Link-type addresses can have the same subnet address even though they are not in the same scope
    // For Global type, this field is 0, so a comparison wouldn't matter
    same &= (local_addr->sin6_scope_id == remote_addr.sin6_scope_id);
    return same;
  } else {
    WARN("Net : Unsupported address family type");
    return false;
  }
}

// 被ncclSocketListen()和findInterfaces()调用
static int findInterfaceMatchSubnet(char* ifNames, 
                                    union socketAddress* localAddrs, 
                                    union socketAddress remoteAddr, 
                                    int ifNameMaxSize, 
                                    int maxIfs) {
  char line[1024], line_a[1024];
  int found = 0;
  struct ifaddrs *interfaces, *interface;
  getifaddrs(&interfaces);
  for (interface = interfaces; interface && !found; interface = interface->ifa_next) {
    if (interface->ifa_addr == NULL) continue;

    /* We only support IPv4 & IPv6 */
    int family = interface->ifa_addr->sa_family;
    if (family != AF_INET && family != AF_INET6)
      continue;

    // check against user specified interfaces
    if (!matchSubnet(*interface, remoteAddr)) {
      continue;
    }

    // Store the local IP address
    int salen = (family == AF_INET) ? sizeof(sockaddr_in) : sizeof(sockaddr_in6);
    memcpy(localAddrs+found, interface->ifa_addr, salen);

    // Store the interface name
    strncpy(ifNames+found*ifNameMaxSize, interface->ifa_name, ifNameMaxSize);

    INFO(NCCL_INIT|NCCL_NET,"NET : Found interface %s:%s in the same subnet as remote address %s", interface->ifa_name, socketToString(&(localAddrs[found].sa), line), socketToString(&(remoteAddr.sa), line_a));
    found++;
    if (found == maxIfs) break;
  }

  if (found == 0) {
    WARN("Net : No interface found in the same subnet as remote address %s", socketToString(&(remoteAddr.sa), line_a));
  }
  freeifaddrs(interfaces);
  return found;
}

// 输入ip_port_pair字符串，输出socketAddr，被findInterfaces()调用
static ncclResult_t GetSocketAddrFromString(union socketAddress* ua, const char* ip_port_pair) {
  if (!(ip_port_pair && strlen(ip_port_pair) > 1)) {
    WARN("Net : string is null");
    return ncclInvalidArgument;
  }

  bool ipv6 = ip_port_pair[0] == '[';
  /* Construct the sockaddress structure */
  if (!ipv6) {
    struct netIf ni;
    // parse <ip_or_hostname>:<port> string, expect one pair
    // ip_port_pair中存储的可能是ip或者主机名
    if (parseStringList(ip_port_pair, &ni, 1) != 1) {
      WARN("Net : No valid <IPv4_or_hostname>:<port> pair found");
      return ncclInvalidArgument;
    }

    struct addrinfo hints, *p;
    int rv;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;  // 协议无关
    hints.ai_socktype = SOCK_STREAM;

    if ( (rv = getaddrinfo(ni.prefix, NULL, &hints, &p)) != 0) {
      WARN("Net : error encountered when getting address info : %s", gai_strerror(rv));
      return ncclInvalidArgument;
    }

    // use the first
    if (p->ai_family == AF_INET) {
      struct sockaddr_in& sin = ua->sin;
      memcpy(&sin, p->ai_addr, sizeof(struct sockaddr_in));
      sin.sin_family = AF_INET;                        // IPv4
      //inet_pton(AF_INET, ni.prefix, &(sin.sin_addr));  // IP address
      sin.sin_port = htons(ni.port);                   // port
    } else if (p->ai_family == AF_INET6) {
      struct sockaddr_in6& sin6 = ua->sin6;
      memcpy(&sin6, p->ai_addr, sizeof(struct sockaddr_in6));
      sin6.sin6_family = AF_INET6;                     // IPv6
      sin6.sin6_port = htons(ni.port);                 // port
      sin6.sin6_flowinfo = 0;                          // needed by IPv6, but possibly obsolete
      sin6.sin6_scope_id = 0;                          // should be global scope, set to 0
    } else {
      WARN("Net : unsupported IP family");
      return ncclInvalidArgument;
    }

    freeaddrinfo(p); // all done with this structure

  } else {
    int i, j = -1, len = strlen(ip_port_pair);
    for (i = 1; i < len; i++) {
      if (ip_port_pair[i] == '%') j = i;
      if (ip_port_pair[i] == ']') break;
    }
    if (i == len) {
      WARN("Net : No valid [IPv6]:port pair found");
      return ncclInvalidArgument;
    }
    bool global_scope = (j == -1 ? true : false);     // If no % found, global scope; otherwise, link scope

    char ip_str[NI_MAXHOST], port_str[NI_MAXSERV], if_name[IFNAMSIZ];
    memset(ip_str, '\0', sizeof(ip_str));
    memset(port_str, '\0', sizeof(port_str));
    memset(if_name, '\0', sizeof(if_name));
    strncpy(ip_str, ip_port_pair+1, global_scope ? i-1 : j-1);
    strncpy(port_str, ip_port_pair+i+2, len-i-1);
    int port = atoi(port_str);
    if (!global_scope) strncpy(if_name, ip_port_pair+j+1, i-j-1); // If not global scope, we need the intf name

    struct sockaddr_in6& sin6 = ua->sin6;
    sin6.sin6_family = AF_INET6;                       // IPv6
    inet_pton(AF_INET6, ip_str, &(sin6.sin6_addr));    // IP address
    sin6.sin6_port = htons(port);                      // port
    sin6.sin6_flowinfo = 0;                            // needed by IPv6, but possibly obsolete
    sin6.sin6_scope_id = global_scope ? 0 : if_nametoindex(if_name);  // 0 if global scope; intf index if link scope
  }
  return ncclSuccess;
}
// 被ncclSocketInit()调用
static int findInterfaces(char* ifNames, union socketAddress *ifAddrs, int ifNameMaxSize, int maxIfs) {  
  int nIfs = 0;  // 记录interface总数
  // Allow user to force the INET socket family selection
  int sock_family = envSocketFamily();
  // User specified interface
  /**
   * NCCL_SOCKET_IFNAME: Define to a list of prefixes to filter interfaces to be used by NCCL. For example, 
   * eth,ib would only select interfaces starting with eth or ib. Using the ^ symbol, NCCL will exclude 
   * interfaces starting with any prefix in that list. For example, ^eth,ib would select interfaces not 
   * starting with eth or ib. Note: By default, the loopback interface (lo) and docker interfaces (docker*) 
   * would not be selected unless there are no other interfaces available. If you prefer to use lo or docker* 
   * over other interfaces, you would need to explicitly select them using NCCL_SOCKET_IFNAME.
   */
  // 利用NCCL_SOKCET_INNAME环境变量过滤interface name
  char* env = getenv("NCCL_SOCKET_IFNAME");
  if (env && strlen(env) > 1) {
    // Specified by user : find or fail  
    // 用户指定的情况下，优先选择用户指定的网卡，若未指定或者指定的网卡未找到再寻找其他的网卡
    nIfs = findInterfaces(env, ifNames, ifAddrs, sock_family, ifNameMaxSize, maxIfs);
  } else {
    // 当用户指定模式未成功时，先查找是否有可用IB网卡，然后再查找同一子网下网卡，最后寻找docker lo等网卡
    // Try to automatically pick the right one
    // Start with IB
    nIfs = findInterfaces("ib", ifNames, ifAddrs, sock_family, ifNameMaxSize, maxIfs);
    // else see if we can get some hint from COMM ID
    if (nIfs == 0) {
      char* commId = getenv("NCCL_COMM_ID");  // ip_port_pair  ipv4 or ipv6
      if (commId && strlen(commId) > 1) {
        // Try to find interface that is in the same subnet as the IP in comm id
        // 从子网中查找
        union socketAddress idAddr;
        GetSocketAddrFromString(&idAddr, commId);
        nIfs = findInterfaceMatchSubnet(ifNames, ifAddrs, idAddr, ifNameMaxSize, maxIfs);
      }
    }
    // Then look for anything else (but not docker or lo)

    if (nIfs == 0) nIfs = findInterfaces("^docker,lo", ifNames, ifAddrs, sock_family, ifNameMaxSize, maxIfs);
    // Finally look for docker, then lo.
    if (nIfs == 0) nIfs = findInterfaces("docker", ifNames, ifAddrs, sock_family, ifNameMaxSize, maxIfs);
    if (nIfs == 0) nIfs = findInterfaces("lo", ifNames, ifAddrs, sock_family, ifNameMaxSize, maxIfs);
  }
  return nIfs;
}

// create listen socket，输入localAddr，返回fd  被ncclSocketListen()调用
static ncclResult_t createListenSocket(int *fd, union socketAddress *localAddr) {
  /* IPv4/IPv6 support */
  int family = localAddr->sa.sa_family;
  int salen = (family == AF_INET) ? sizeof(sockaddr_in) : sizeof(sockaddr_in6);

  /* Create socket and bind it to a port */
  int sockfd = socket(family, SOCK_STREAM, 0);
  if (sockfd == -1) {
    WARN("Net : Socket creation failed : %s", strerror(errno));
    return ncclSystemError;
  }

  if (socketToPort(&localAddr->sa)) {
    // Port is forced by env. Make sure we get the port.
    int opt = 1;
    SYSCHECK(setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt)), "setsockopt");
  }

  // localAddr port should be 0 (Any port)
  SYSCHECK(bind(sockfd, &localAddr->sa, salen), "bind");

  /* Get the assigned Port */
  socklen_t size = salen;
  SYSCHECK(getsockname(sockfd, &localAddr->sa, &size), "getsockname");

#ifdef ENABLE_TRACE
  char line[1024];
  TRACE(NCCL_INIT|NCCL_NET,"Listening on socket %s", socketToString(&localAddr->sa, line));
#endif

  /* Put the socket in listen mode
   * NB: The backlog will be silently truncated to the value in /proc/sys/net/core/somaxconn
   */
  SYSCHECK(listen(sockfd, 16384), "listen");
  *fd = sockfd;
  return ncclSuccess;
}

// 先创建一个本地socket，然后将创建的本地socket连接到remoteAddr  被ncclSocketConnect()调用
static ncclResult_t connectAddress(int* fd, union socketAddress* remoteAddr) {
  /* IPv4/IPv6 support */
  int family = remoteAddr->sa.sa_family;
  int salen = (family == AF_INET) ? sizeof(sockaddr_in) : sizeof(sockaddr_in6);

  /* Connect to a hostname / port */
  *fd = socket(family, SOCK_STREAM, 0);
  if (*fd == -1) {
    WARN("Net : Socket creation failed : %s", strerror(errno));
    return ncclSystemError;
  }

  const int one = 1;
  SYSCHECK(setsockopt(*fd, IPPROTO_TCP, TCP_NODELAY, (char*)&one, sizeof(int)), "setsockopt");

  /*  const int bufsize = 128*1024;
    SYSCHECK(setsockopt(*fd, SOL_SOCKET, SO_SNDBUF, (char*)&bufsize, sizeof(int)), "setsockopt");
    SYSCHECK(setsockopt(*fd, SOL_SOCKET, SO_RCVBUF, (char*)&bufsize, sizeof(int)), "setsockopt");*/

  char line[1024];
#ifdef ENABLE_TRACE
  TRACE(NCCL_INIT|NCCL_NET,"Connecting to socket %s", socketToString(&remoteAddr->sa, line));
#endif

  int ret;
  int timedout_retries = 0;
  int refused_retries = 0;
retry:
  SYSCHECKSYNC(connect(*fd, &remoteAddr->sa, salen), "connect", ret);  // connect()函数成功返回0，失败返回-1
  if (ret == 0) return ncclSuccess;
  if ((errno == ECONNREFUSED || errno == ETIMEDOUT)) {
    if ((errno == ECONNREFUSED && ++refused_retries < RETRY_REFUSED_TIMES) ||
        (errno == ETIMEDOUT && ++timedout_retries < RETRY_TIMEDOUT_TIMES)) {
      INFO(NCCL_ALL,"Call to connect returned %s, retrying", strerror(errno));
      usleep(SLEEP_INT);
      goto retry;
    }
  }
  WARN("Connect to %s failed : %s", socketToString(&remoteAddr->sa, line), strerror(errno));
  return ncclSystemError;
}

#define NCCL_SOCKET_SEND 0
#define NCCL_SOCKET_RECV 1
// 利用recv()和send()函数实现socket的不同操作  被ncclSocketTest()调用
static ncclResult_t socketProgress(int op, int fd, void* ptr, int size, int* offset) {
  int bytes = 0;
  char* data = (char*)ptr;
  do {
    // 返回值 <0 出错   =0 连接关闭   >0 接收到数据大小
    if (op == NCCL_SOCKET_RECV) bytes = recv(fd, data+(*offset), size-(*offset), MSG_DONTWAIT);  // offset最初始为0，后逐渐累加
    if (op == NCCL_SOCKET_SEND) bytes = send(fd, data+(*offset), size-(*offset), MSG_DONTWAIT);
    if (op == NCCL_SOCKET_RECV && bytes == 0) {
      WARN("Net : Connection closed by remote peer");
      return ncclSystemError;
    }
    if (bytes == -1) {
      if (errno != EINTR && errno != EWOULDBLOCK && errno != EAGAIN) {
        WARN("Call to recv failed : %s", strerror(errno));
        return ncclSystemError;
      } else {
        bytes = 0;
      }
    }
    (*offset) += bytes;
  } while (bytes > 0 && (*offset) < size);  // 多次recv/send，知道所有的数据被recv/send完毕
  return ncclSuccess;
}

// 利用socketProgress()实现socketWait()  被ncclSocketTest()调用
static ncclResult_t socketWait(int op, int fd, void* ptr, int size, int* offset) {
  while (*offset < size)
    NCCLCHECK(socketProgress(op, fd, ptr, size, offset));
  return ncclSuccess;
}

// 利用socketProgress()实现socketSend()  被ncclSocketTest()调用
static ncclResult_t socketSend(int fd, void* ptr, int size) {
  int offset = 0;
  NCCLCHECK(socketWait(NCCL_SOCKET_SEND, fd, ptr, size, &offset));
  return ncclSuccess;
}

// 利用socketProgress()实现socketReceive()  被ncclSocketTest()调用
static ncclResult_t socketReceive(int fd, void* ptr, int size) {
  int offset = 0;
  NCCLCHECK(socketWait(NCCL_SOCKET_RECV, fd, ptr, size, &offset));
  return ncclSuccess;
}

#endif
