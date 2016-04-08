cmake_minimum_required(VERSION 3.4)
project(ParallelComputingUNAL)

find_package(Threads REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp)
add_executable(ParallelComputingUNAL ${SOURCE_FILES})

add_executable(PIFork PIFork.cpp)

add_executable(PIThreads PIThreads.cpp)
target_link_libraries(PIThreads ${CMAKE_THREAD_LIBS_INIT})