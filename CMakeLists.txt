cmake_minimum_required(VERSION 3.16)
project(ant_alg_mpi)

set(CMAKE_CXX_STANDARD 20)

include_directories($ENV{MSMPI_INC})
link_directories($ENV{MSMPI_LIB32})

add_executable(ant_alg_mpi main.cpp Ant.cpp Ant.h AntAlg.cpp AntAlg.h)
target_link_libraries(ant_alg_mpi msmpi)