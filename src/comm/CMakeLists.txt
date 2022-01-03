project(COMM)

# aux_source_directory(. max_src)

find_package(Threads REQUIRED)


file(GLOB CXX_SRC
  comm_manager.cpp
  comm_manager.hpp
)

add_library(comm_LIB STATIC ${CXX_SRC})
target_link_libraries(comm_LIB pthread)

add_executable(server server.cpp)
target_link_libraries(server comm_LIB)

add_executable(client client.cpp)
target_link_libraries(client comm_LIB)