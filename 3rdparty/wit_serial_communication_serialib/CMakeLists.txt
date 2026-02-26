cmake_minimum_required(VERSION 3.10)
project(serial_communication)
set(CMAKE_CXX_STANDARD 11)

#[[
# Pangolin
set(Pangolin_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/Pangolin-master/build/install/lib/cmake/Pangolin)
find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

# glew
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/glew-2.2.0/include)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/glew-2.2.0/lib/Release/x64)

#eigen

#Yesense
#target_link_libraries(Yesense PUBLIC serial_lib ${Pangolin_LIBRARIES} pthread decode psins)
]]
# 添加Serialib库的头文件和源文件
include_directories(${CMAKE_SOURCE_DIR}/include)

# 添加源文件
add_executable(serial_communication
    src/main.cpp
    src/Serialib.cpp
)
