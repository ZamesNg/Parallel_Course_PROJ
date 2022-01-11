# add_subdirectory(hello)

add_subdirectory(utils)
add_subdirectory(max)
add_subdirectory(comm)
add_subdirectory(sum)
add_subdirectory(sort)
add_subdirectory(class_hw)

project(main)

file(GLOB CXX_SRC main.cpp)

FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}${OpenMP_EXE_LINKER_FLAGS}")
endif()

FIND_PACKAGE(CUDA REQUIRED)
if (NOT CUDA_FOUND)
  message(STATUS "CUDA not found. Project will not be built.")
endif(NOT CUDA_FOUND)

SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_61,code=sm_61;)
include_directories("${CUDA_INCLUDE_DIRS}")

include_directories(${CMAKE_SOURCE_DIR}/src)

add_executable(${PROJECT_NAME} ${CXX_SRC})
target_link_libraries(${PROJECT_NAME} 
  sort_CUDA
  sum_CUDA
  max_CUDA
  common_CUDA
  sort_LIB
  sum_LIB
  max_LIB
  comm_LIB
  common_LIB)
