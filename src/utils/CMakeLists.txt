project(common)

# aux_source_directory(. max_src)

FIND_PACKAGE(CUDA REQUIRED)
if (NOT CUDA_FOUND)
  message(STATUS "CUDA not found. Project will not be built.")
endif(NOT CUDA_FOUND)

SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_61,code=sm_61;)
include_directories("${CUDA_INCLUDE_DIRS}")


# build a cuda library
file(GLOB CUDA_SRC
  ./*.cu
  ./*.cuh
)
cuda_add_library(${PROJECT_NAME}_CUDA SHARED ${CUDA_SRC})

file(GLOB CXX_SRC
  ${PROJECT_NAME}.cpp
  ${PROJECT_NAME}.hpp
)

add_library(${PROJECT_NAME}_LIB STATIC ${CXX_SRC})