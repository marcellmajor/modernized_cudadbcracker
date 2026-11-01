cmake_minimum_required(VERSION 3.24)
project(cudadbcracker LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(CUDADB_ENABLE_CUDA "Build CUDA kernels (requires a CUDA toolkit)" ON)

add_library(cudadb_host STATIC
  src/types.cpp
  src/reference_sha1.cpp
  src/enumeration.cpp
  src/input.cpp
  src/checkpoint.cpp
  src/tuning.cpp
  src/search.cpp
)
target_include_directories(cudadb_host PUBLIC include)
target_compile_options(cudadb_host PRIVATE -Wall -Wextra)

if(CUDADB_ENABLE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "75-real;80-real;89-real;90-real;90-virtual")
  endif()
  add_library(cudadb_cuda STATIC src/cuda/kernels.cu src/cuda/dispatch.cu src/cuda/device_info.cu)
  target_include_directories(cudadb_cuda PRIVATE src/cuda include)
  target_link_libraries(cudadb_cuda PUBLIC cudadb_host)
  target_compile_options(cudadb_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo -Xptxas=-v>)
  set(CUDADB_DISPATCH_SRC "")
else()
  set(CUDADB_DISPATCH_SRC src/dispatch_stub.cpp)
endif()

add_executable(cudadbcracker src/main.cpp src/cli.cpp ${CUDADB_DISPATCH_SRC})
target_include_directories(cudadbcracker PRIVATE include)
if(CUDADB_ENABLE_CUDA)
  target_link_libraries(cudadbcracker PRIVATE cudadb_cuda)
else()
  target_link_libraries(cudadbcracker PRIVATE cudadb_host)
endif()

include(CTest)
if(BUILD_TESTING)
  foreach(t test_host test_search test_resume)
    add_executable(${t} tests/${t}.cpp)
    target_link_libraries(${t} PRIVATE cudadb_host)
    target_compile_definitions(${t} PRIVATE FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures")
    add_test(NAME ${t} COMMAND ${t})
    set_tests_properties(${t} PROPERTIES LABELS host)
  endforeach()
  add_executable(test_gpu tests/test_gpu.cpp ${CUDADB_DISPATCH_SRC})
  if(CUDADB_ENABLE_CUDA)
    target_link_libraries(test_gpu PRIVATE cudadb_cuda)
  else()
    target_link_libraries(test_gpu PRIVATE cudadb_host)
  endif()
  add_test(NAME test_gpu COMMAND test_gpu)
  set_tests_properties(test_gpu PROPERTIES LABELS gpu SKIP_RETURN_CODE 77)
endif()
