cmake_minimum_required(VERSION 3.10.2)
project("opencl_cpp_runner")

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCL REQUIRED)

include_directories(
        common/
        implementations/
        ${OpenCL_INCLUDE_DIR}
)

file(GLOB src_files
        main.cpp
        common/*.cpp
        implementations/*.cpp
)

add_executable(${CMAKE_PROJECT_NAME}
    ${src_files}
)

target_link_libraries(${CMAKE_PROJECT_NAME}
    ${OpenCL_LIBRARIES}
)
