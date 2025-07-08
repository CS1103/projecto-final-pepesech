cmake_minimum_required(VERSION 3.10)
project(ProyectoFinal)

set(CMAKE_CXX_STANDARD 20)

add_executable(proyecto src/main.cpp)
add_executable(generacion tools/gen_dataset.cpp)