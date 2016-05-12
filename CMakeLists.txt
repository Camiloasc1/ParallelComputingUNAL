cmake_minimum_required(VERSION 3.4)
project(ParallelComputingUNAL)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")

find_package(Threads)

find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()

find_package(CUDA)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52 --compiler-options -std=c++98")
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

set(SOURCE_FILES main.cpp)
add_executable(ParallelComputingUNAL ${SOURCE_FILES})

add_executable(PIFork PI/PIFork.c)

add_executable(PIThreads PI/PIThreads.c)
target_link_libraries(PIThreads ${CMAKE_THREAD_LIBS_INIT})

add_executable(PIOpenMP PI/PIOpenMP.c)

add_executable(PIOpenMPFalseSharing PI/PIOpenMPFalseSharing.c)

add_executable(MatrixFork Matrix/MatrixFork.c)
#target_link_libraries(MatrixFork m)

add_executable(MatrixThreads Matrix/MatrixThreads.c)
target_link_libraries(MatrixThreads ${CMAKE_THREAD_LIBS_INIT})

add_executable(MatrixOpenMP Matrix/MatrixOpenMP.c)

cuda_add_executable(MatrixCUDA Matrix/MatrixCUDA.cu)

add_executable(LeibnizSequential PI/FinalExam/Leibniz/Sequential.c)
add_executable(LeibnizOMP PI/FinalExam/Leibniz/OMP.c)
cuda_add_executable(LeibnizCUDA PI/FinalExam/Leibniz/CUDA.cu)
target_link_libraries(LeibnizSequential m)
target_link_libraries(LeibnizOMP m)
target_link_libraries(LeibnizCUDA m)

add_executable(GaussLegendreSequential PI/FinalExam/GaussLegendre/Sequential.c)
add_executable(GaussLegendreOMP PI/FinalExam/GaussLegendre/OMP.c)
cuda_add_executable(GaussLegendreCUDA PI/FinalExam/GaussLegendre/CUDA.cu)
target_link_libraries(GaussLegendreSequential m)
target_link_libraries(GaussLegendreOMP m)
target_link_libraries(GaussLegendreCUDA m)

add_executable(MontecarloSequential PI/FinalExam/Montecarlo/Sequential.c)
add_executable(MontecarloOMP PI/FinalExam/Montecarlo/OMP.c)
cuda_add_executable(MontecarloCUDA PI/FinalExam/Montecarlo/CUDA.cu)
target_link_libraries(MontecarloSequential m)
target_link_libraries(MontecarloOMP m)
target_link_libraries(MontecarloCUDA ${CUDA_curand_LIBRARY} m)