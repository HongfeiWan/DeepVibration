#ifndef SERIALIB_H
#define SERIALIB_H

#include <string>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>

#ifdef _WIN32
    #include <windows.h>
#else
    #include <termios.h>
    #include <sys/ioctl.h>
    #include <sys/time.h>
    #include <sys/types.h>
    #include <sys/stat.h>
    #include <fcntl.h>
    #include <unistd.h>
    #include <errno.h>
    #include <cstring>
#endif

class Serialib {
public:
    Serialib();
    ~Serialib();
    
    bool open(const char* portName, unsigned long baudRate);
    void close();
    bool isOpen() const;
    int read(unsigned char* buffer, size_t bufferSize, unsigned long timeoutMs);
    int write(const unsigned char* buffer, size_t bufferSize);
    void setTimeout(unsigned long timeoutMs);
    void setBaudRate(unsigned long baudRate);
    
private:
    #ifdef _WIN32
        HANDLE hSerial;
        COMMTIMEOUTS timeouts;
    #else
        int fd;
        struct termios options;
    #endif
    
    unsigned long timeoutMs;
};

#endif // SERIALIB_H