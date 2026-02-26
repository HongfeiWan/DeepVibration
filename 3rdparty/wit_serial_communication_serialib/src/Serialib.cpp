#include "Serialib.h"

Serialib::Serialib() : timeoutMs(1000) {
    #ifdef _WIN32
        hSerial = INVALID_HANDLE_VALUE;
    #else
        fd = -1;
    #endif
}

Serialib::~Serialib() {
    close();
}

bool Serialib::open(const char* portName, unsigned long baudRate) {
    #ifdef _WIN32
        hSerial = CreateFileA(
            portName,
            GENERIC_READ | GENERIC_WRITE,
            0,
            NULL,
            OPEN_EXISTING,
            FILE_ATTRIBUTE_NORMAL,
            NULL
        );
        
        if (hSerial == INVALID_HANDLE_VALUE) {
            return false;
        }
        
        DCB dcb = {0};
        dcb.DCBlength = sizeof(dcb);
        if (!GetCommState(hSerial, &dcb)) {
            CloseHandle(hSerial);
            return false;
        }
        
        dcb.BaudRate = baudRate;
        dcb.ByteSize = 8;
        dcb.StopBits = ONESTOPBIT;
        dcb.Parity = NOPARITY;
        dcb.fDtrControl = DTR_CONTROL_ENABLE;
        
        if (!SetCommState(hSerial, &dcb)) {
            CloseHandle(hSerial);
            return false;
        }
        
        timeouts.ReadIntervalTimeout = MAXDWORD;
        timeouts.ReadTotalTimeoutConstant = timeoutMs;
        timeouts.ReadTotalTimeoutMultiplier = 0;
        timeouts.WriteTotalTimeoutConstant = timeoutMs;
        timeouts.WriteTotalTimeoutMultiplier = 0;
        
        if (!SetCommTimeouts(hSerial, &timeouts)) {
            CloseHandle(hSerial);
            return false;
        }
    #else
        fd = ::open(portName, O_RDWR | O_NOCTTY | O_NDELAY);
        if (fd < 0) {
            return false;
        }
        
        if (::ioctl(fd, FIONBIO, (unsigned long)1) < 0) {
            ::close(fd);
            return false;
        }
        
        struct termios newOpt;
        bzero(&newOpt, sizeof(newOpt));
        newOpt.c_cflag = baudRate | CS8 | CLOCAL | CREAD;
        newOpt.c_iflag = IGNPAR;
        newOpt.c_oflag = 0;
        newOpt.c_lflag = 0;
        newOpt.c_cc[VTIME] = 0;
        newOpt.c_cc[VMIN] = 0;
        
        if (::tcflush(fd, TCIFLUSH) < 0 || ::tcsetattr(fd, TCSANOW, &newOpt) < 0) {
            ::close(fd);
            return false;
        }
    #endif
    
    return true;
}

void Serialib::close() {
    #ifdef _WIN32
        if (hSerial != INVALID_HANDLE_VALUE) {
            CloseHandle(hSerial);
            hSerial = INVALID_HANDLE_VALUE;
        }
    #else
        if (fd >= 0) {
            ::close(fd);
            fd = -1;
        }
    #endif
}

bool Serialib::isOpen() const {
    #ifdef _WIN32
        return hSerial != INVALID_HANDLE_VALUE;
    #else
        return fd >= 0;
    #endif
}

int Serialib::read(unsigned char* buffer, size_t bufferSize, unsigned long timeoutMs) {
    if (!isOpen()) return -1;
    
    #ifdef _WIN32
        DWORD bytesRead = 0;
        if (!ReadFile(hSerial, buffer, bufferSize, &bytesRead, NULL)) {
            return -1;
        }
        return bytesRead;
    #else
        fd_set readfds;
        struct timeval timeout;
        FD_ZERO(&readfds);
        FD_SET(fd, &readfds);
        timeout.tv_sec = timeoutMs / 1000;
        timeout.tv_usec = (timeoutMs % 1000) * 1000;
        
        int result = ::select(fd + 1, &readfds, NULL, NULL, &timeout);
        if (result <= 0) {
            return result;
        }
        
        int bytesRead = ::read(fd, buffer, bufferSize);
        return bytesRead;
    #endif
}

int Serialib::write(const unsigned char* buffer, size_t bufferSize) {
    if (!isOpen()) return -1;
    
    #ifdef _WIN32
        DWORD bytesWritten = 0;
        if (!WriteFile(hSerial, buffer, bufferSize, &bytesWritten, NULL)) {
            return -1;
        }
        return bytesWritten;
    #else
        return ::write(fd, buffer, bufferSize);
    #endif
}

void Serialib::setTimeout(unsigned long timeoutMs) {
    this->timeoutMs = timeoutMs;
    #ifdef _WIN32
        if (isOpen()) {
            timeouts.ReadTotalTimeoutConstant = timeoutMs;
            timeouts.WriteTotalTimeoutConstant = timeoutMs;
            SetCommTimeouts(hSerial, &timeouts);
        }
    #endif
}

void Serialib::setBaudRate(unsigned long baudRate) {
    #ifdef _WIN32
        if (isOpen()) {
            DCB dcb = {0};
            dcb.DCBlength = sizeof(dcb);
            if (GetCommState(hSerial, &dcb)) {
                dcb.BaudRate = baudRate;
                SetCommState(hSerial, &dcb);
            }
        }
    #else
        if (isOpen()) {
            struct termios options;
            ::tcgetattr(fd, &options);
            ::cfsetspeed(&options, baudRate);
            ::tcsetattr(fd, TCSANOW, &options);
        }
    #endif
}