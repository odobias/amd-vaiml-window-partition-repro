cmake_minimum_required(VERSION 3.20)
project(amd_vaiml_window_partition_repro LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT ORT_INCLUDE_DIR)
  set(ORT_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime/include")
endif()

if(NOT EXISTS "${ORT_INCLUDE_DIR}/onnxruntime_c_api.h")
  message(FATAL_ERROR "onnxruntime_c_api.h not found in ORT_INCLUDE_DIR=${ORT_INCLUDE_DIR}. Run bootstrap.ps1 first or pass -DORT_INCLUDE_DIR=<path>.")
endif()

add_executable(amd_vaiml_repro src/main.cpp)
target_include_directories(amd_vaiml_repro PRIVATE "${ORT_INCLUDE_DIR}")
target_compile_definitions(amd_vaiml_repro PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
