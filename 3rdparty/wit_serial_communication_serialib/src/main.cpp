#include <cstdint>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <ctime>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include "Serialib.h"

// CRC-16(Modbus)校验计算
uint16_t calculateCRC16(const std::vector<unsigned char>& data) {
    uint16_t crc = 0xFFFF;
    for (auto byte : data) {
        crc ^= byte;
        for (int i = 0; i < 8; i++) {
            if (crc & 0x0001) {
                crc >>= 1;
                crc ^= 0xA001;
            } else {
                crc >>= 1;
            }
        }
    }
    return crc;
}

// 将十六进制字符串转换为字节数组
std::vector<unsigned char> hexStringToBytes(const std::string& hexStr) {
    std::vector<unsigned char> bytes;
    for (size_t i = 0; i < hexStr.length(); i += 2) {
        std::string byteStr = hexStr.substr(i, 2);
        char byte = (char) strtol(byteStr.c_str(), nullptr, 16);
        bytes.push_back(static_cast<unsigned char>(byte));
    }
    return bytes;
}

// 获取当前系统时间（yyyy-MM-dd HH:mm:ss.sss格式）
std::string getCurrentTimeStr() {
    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&now_c), "%Y-%m-%d %H:%M:%S");
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    ss << '.' << std::setw(3) << std::setfill('0') << ms.count();
    return ss.str();
}

// 获取当前日期（yyyy-MM-dd格式）
std::string getCurrentDateStr() {
    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&now_c), "%Y-%m-%d");
    return ss.str();
}

// 读取加速度
void readAcceleration(Serialib& serial, std::vector<std::string>& dataRow) {
    std::string hexData = "5003003400034984";
    std::vector<unsigned char> dataToSend = hexStringToBytes(hexData);
    if (serial.write(dataToSend.data(), dataToSend.size()) != dataToSend.size()) {
        std::cerr << "SEND ACCELERATION FAILED" << std::endl;
        return;
    }
    std::vector<unsigned char> receivedData(1024, 0);
    int bytesRead = serial.read(receivedData.data(), receivedData.size(), 0);

    if (bytesRead > 0) {
        receivedData.resize(bytesRead);
        if (bytesRead >= 10) {
            int16_t AX = ((int16_t)receivedData[3] << 8) | receivedData[4];
            int16_t AY = ((int16_t)receivedData[5] << 8) | receivedData[6];
            int16_t AZ = ((int16_t)receivedData[7] << 8) | receivedData[8];
            dataRow.push_back(std::to_string((AX * 16.0000) / 32768));
            dataRow.push_back(std::to_string((AY * 16.0000) / 32768));
            dataRow.push_back(std::to_string((AZ * 16.0000) / 32768));
        } else {
            std::cerr << "ACCELERATION DATA FORMAT ERROR" << std::endl;
        }
    } else {
        std::cerr << "ACCELERATION READ FAILED" << std::endl;
    }
}

// 读取速度
void readVelocity(Serialib& serial, std::vector<std::string>& dataRow) {
    std::string hexData = "5003003A00032847";
    std::vector<unsigned char> dataToSend = hexStringToBytes(hexData);

    if (serial.write(dataToSend.data(), dataToSend.size()) != dataToSend.size()) {
        std::cerr << "SEND VELOCITY FAILED" << std::endl;
        return;
    }

    std::vector<unsigned char> receivedData(1024, 0);
    int bytesRead = serial.read(receivedData.data(), receivedData.size(), 0);

    if (bytesRead > 0) {
        receivedData.resize(bytesRead);
        if (bytesRead >= 10) {
            int16_t VX = ((int16_t)receivedData[3] << 8) | receivedData[4];
            int16_t VY = ((int16_t)receivedData[5] << 8) | receivedData[6];
            int16_t VZ = ((int16_t)receivedData[7] << 8) | receivedData[8];
            dataRow.push_back(std::to_string(VX / 100.00));
            dataRow.push_back(std::to_string(VY / 100.00));
            dataRow.push_back(std::to_string(VZ / 100.00));
        } else {
            std::cerr << "VELOCITY DATA FORMAT ERROR" << std::endl;
        }
    } else {
        std::cerr << "VELOCITY READ FAILED" << std::endl;
    }
}

// 读取温度
void readTemperature(Serialib& serial, std::vector<std::string>& dataRow) {
    std::string hexData = "500300400001885F";
    std::vector<unsigned char> dataToSend = hexStringToBytes(hexData);

    if (serial.write(dataToSend.data(), dataToSend.size()) != dataToSend.size()) {
        std::cerr << "SEND TEMPERATURE FAILED" << std::endl;
        return;
    }

    std::vector<unsigned char> receivedData(1024, 0);
    int bytesRead = serial.read(receivedData.data(), receivedData.size(), 0);

    if (bytesRead > 0) {
        receivedData.resize(bytesRead);
        if (bytesRead >= 6) {
            int16_t temp = ((int16_t)receivedData[3] << 8) | receivedData[4];
            dataRow.push_back(std::to_string(temp / 100.0));
        } else {
            std::cerr << "TEMPERATURE DATA FORMAT ERROR" << std::endl;
        }
    } else {
        std::cerr << "TEMPERATURE READ FAILED" << std::endl;
    }
}

// 读取震动位移
void readDisplacement(Serialib& serial, std::vector<std::string>& dataRow) {
    std::string hexData = "500300410003585E";
    std::vector<unsigned char> dataToSend = hexStringToBytes(hexData);

    if (serial.write(dataToSend.data(), dataToSend.size()) != dataToSend.size()) {
        std::cerr << "SEND DISPLACEMENT FAILED" << std::endl;
        return;
    }

    std::vector<unsigned char> receivedData(1024, 0);
    int bytesRead = serial.read(receivedData.data(), receivedData.size(), 0);

    if (bytesRead > 0) {
        receivedData.resize(bytesRead);
        if (bytesRead >= 10) {
            int16_t DX = ((int16_t)receivedData[3] << 8) | receivedData[4];
            int16_t DY = ((int16_t)receivedData[5] << 8) | receivedData[6];
            int16_t DZ = ((int16_t)receivedData[7] << 8) | receivedData[8];
            dataRow.push_back(std::to_string(DX * 100.00 / 10.00));
            dataRow.push_back(std::to_string(DY * 100.00 / 10.00));
            dataRow.push_back(std::to_string(DZ * 100.00 / 10.00));
        } else {
            std::cerr << "DISPLACEMENT DATA FORMAT ERROR" << std::endl;
        }
    } else {
        std::cerr << "DISPLACEMENT READ FAILED" << std::endl;
    }
}

// 读取震动频率
void readFrequency(Serialib& serial, std::vector<std::string>& dataRow) {
    std::string hexData = "500300440003485F";
    std::vector<unsigned char> dataToSend = hexStringToBytes(hexData);

    if (serial.write(dataToSend.data(), dataToSend.size()) != dataToSend.size()) {
        std::cerr << "SEND FREQUENCY FAILED" << std::endl;
        return;
    }

    std::vector<unsigned char> receivedData(1024, 0);
    int bytesRead = serial.read(receivedData.data(), receivedData.size(), 0);

    if (bytesRead > 0) {
        receivedData.resize(bytesRead);
        if (bytesRead >= 10) {
            int16_t FX = ((int16_t)receivedData[3] << 8) | receivedData[4];
            int16_t FY = ((int16_t)receivedData[5] << 8) | receivedData[6];
            int16_t FZ = ((int16_t)receivedData[7] << 8) | receivedData[8];
            dataRow.push_back(std::to_string(FX / 10.0));
            dataRow.push_back(std::to_string(FY / 10.0));
            dataRow.push_back(std::to_string(FZ / 10.0));
        } else {
            std::cerr << "FREQUENCY DATA FORMAT ERROR" << std::endl;
        }
    } else {
        std::cerr << "FREQUENCY READ FAILED" << std::endl;
    }
}

// 读取所有数据
void readAllData(Serialib& serial, std::vector<std::string>& dataRow) {
    dataRow.clear();
    std::string timestamp = getCurrentTimeStr();
    dataRow.push_back(timestamp);
    readTemperature(serial, dataRow);
    readVelocity(serial, dataRow);
    readAcceleration(serial, dataRow);
    readDisplacement(serial, dataRow);
    readFrequency(serial, dataRow);
}

// 读取数据的线程函数
void readDataThread(Serialib& serial, std::vector<std::string>& dataRow, std::mutex& dataMutex, std::condition_variable& dataCond) {
    while (true) {
        std::vector<std::string> localDataRow;
        readAllData(serial, localDataRow);
        {
            std::lock_guard<std::mutex> lock(dataMutex);
            dataRow = localDataRow;
        }
        dataCond.notify_one();
        //std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

// 打印数据到CMD窗口
void printData(const std::vector<std::string>& dataRow, int detectorIndex) {
    std::string timestamp = dataRow[0];
    std::string temperature = dataRow[1];
    std::string velocityX = dataRow[2];
    std::string velocityY = dataRow[3];
    std::string velocityZ = dataRow[4];
    std::string accelerationX = dataRow[5];
    std::string accelerationY = dataRow[6];
    std::string accelerationZ = dataRow[7];
    std::string displacementX = dataRow[8];
    std::string displacementY = dataRow[9];
    std::string displacementZ = dataRow[10];
    std::string frequencyX = dataRow[11];
    std::string frequencyY = dataRow[12];
    std::string frequencyZ = dataRow[13];
    std::cout << "---------------------------------------------" << std::endl;
    std::cout << "Detector " << detectorIndex << ":" << std::endl;
    std::cout << "Timestamp: " << timestamp << std::endl;
    std::cout << "Temperature: " << temperature << " C" << std::endl;
    std::cout << "Velocity: VX: " << velocityX << " mm/s, VY: " << velocityY << " mm/s, VZ: " << velocityZ << " mm/s" << std::endl;
    std::cout << "Acceleration: AX: " << accelerationX << " g * m/(s^2), AY: " << accelerationY << " g * m/(s^2), AZ: " << accelerationZ << " g * m/(s^2)" << std::endl;
    std::cout << "Displacement: DX: " << displacementX << " e-6m, DY: " << displacementY << " e-6m, DZ: " << displacementZ << " e-6m" << std::endl;
    std::cout << "Frequency: FX: " << frequencyX << " Hz, FY: " << frequencyY << " Hz, FZ: " << frequencyZ << " Hz" << std::endl;
}

// 保存数据到文件
void saveDataToFile(const std::vector<std::string>& dataRow, const std::string& filename) {
    std::ofstream file(filename, std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }
    for (const auto& data : dataRow) {
        file << data << ",";
    }
    file << "\n";
    file.close();
}


void saveDataToFileAsync(const std::vector<std::string>& dataRow, const std::string& filename) {
    saveDataToFile(dataRow, filename);
}


int main() {
    int num;
    std::cout << "PLEASE ENTER THE NUMBER OF DETECTORS: ";
    std::cin >> num;
    std::cin.ignore(); // 忽略换行符

    Serialib serial[num];
    std::vector<std::string> dataRow[num];
    std::mutex dataMutex[num];
    std::condition_variable dataCond[num];
    std::thread threads[num];
    std::vector<std::string> filenames(num);
    std::string currentDate = getCurrentDateStr();

    // 手动输入串口名称
    std::string portNames[num];
    for (int i = 0; i < num; ++i) {
        std::cout << "PLEASE ENTER THE PORT FOR DETECTOR " << i + 1 << " (EG. COM3 OR /dev/ttyUSB0):";
        std::getline(std::cin, portNames[i]);
        if (portNames[i].length()>4){
            portNames[i] = "\\\\.\\\\" + portNames[i];
        }
        if (!serial[i].open(portNames[i].c_str(), 230400)) {
            std::cerr << "CANNOT OPEN THE PORT: " << portNames[i] << std::endl;
            return 1;
        }
        std::cout << "THE PORT IS OPENED: " << portNames[i] << std::endl;
        // 生成文件名
        filenames[i] = "detector_" + std::to_string(i + 1) + "_" + currentDate + ".txt";
    }

    // 创建线程
    for (int i = 0; i < num; ++i) {
        threads[i] = std::thread(readDataThread, std::ref(serial[i]), std::ref(dataRow[i]), std::ref(dataMutex[i]), std::ref(dataCond[i]));
    }
    // 死循环：持续读取数据
    while (true) {
        std::vector<std::vector<std::string>> dataRows(num);
        std::vector<std::future<void>> futures(num);
        std::string newDate = getCurrentDateStr();
        if (newDate != currentDate) {
            // 日期变化，更新文件名
            currentDate = newDate;
            for (int i = 0; i < num; ++i) {
                filenames[i] = "detector_" + std::to_string(i + 1) + "_" + currentDate + ".txt";
            }
        }

        for (int i = 0; i < num; ++i) {
            futures[i] = std::async(std::launch::async, [&dataRows, &dataMutex, &dataCond, &dataRow, i] {
                std::unique_lock<std::mutex> lock(dataMutex[i]);
                dataCond[i].wait(lock);
                dataRows[i] = dataRow[i];
            });
        }

        for (int i = 0; i < num; ++i) {
            futures[i].get(); // 等待异步任务完成
            // 异步写入数据
            std::thread saveThread(saveDataToFileAsync, dataRows[i], filenames[i]);
            saveThread.detach(); // 异步写入文件
        }
    }

    // 关闭串口
    for (int i = 0; i < 4; ++i) {
        serial[i].close();
    }

    std::cout << "IO CLOSED" << std::endl;
    return 0;
}
