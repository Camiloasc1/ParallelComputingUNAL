cmake_minimum_required(VERSION 3.4)
project(ParallelComputingUNAL)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")

find_package(Threads REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11")

set(SOURCE_FILES main.cpp)
add_executable(ParallelComputingUNAL ${SOURCE_FILES})

add_executable(PIFork PI/PIFork.c)

add_executable(PIThreads PI/PIThreads.c)
target_link_libraries(PIThreads ${CMAKE_THREAD_LIBS_INIT})

add_executable(MatrixFork Matrix/MatrixFork.c)

add_executable(MatrixThreads Matrix/MatrixThreads.c)
target_link_libraries(PIThreads ${CMAKE_THREAD_LIBS_INIT})