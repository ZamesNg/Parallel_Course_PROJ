project(mystring)

# aux_source_directory(. max_src)

file(GLOB CXX_SRC
  ${PROJECT_NAME}.cpp
  ${PROJECT_NAME}.hpp
)

add_library(${PROJECT_NAME}_LIB STATIC ${CXX_SRC})

include_directories(${CMAKE_SOURCE_DIR}/src)
add_executable(${PROJECT_NAME} ${PROJECT_NAME}_main.cpp)

# link the max with cuda lib
target_link_libraries(${PROJECT_NAME} 
  ${PROJECT_NAME}_LIB)