cmake_minimum_required(VERSION 3.16)
project(rng_cli LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(rng-cli src/main.cpp)

install(TARGETS rng-cli RUNTIME DESTINATION bin)
