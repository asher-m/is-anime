cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(isanime)

find_package(Torch REQUIRED PATHS ../libtorch)

add_executable(isanime isanime.cpp model.cpp frame.cpp)
target_link_libraries(isanime "${TORCH_LIBRARIES}")
