project(COMM)

# aux_source_directory(. max_src)

find_package(Threads REQUIRED)


file(GLOB CXX_SRC
  comm_manager.cpp
  comm_manager.hpp
)

add_library(comm_lib STATIC ${CXX_SRC})
target_link_libraries(comm_lib pthread)

add_executable(server server.cpp)
target_link_libraries(server comm_lib)

add_executable(client client.cpp)
target_link_libraries(client comm_lib)