cmake_minimum_required(VERSION 3.16)
project(statarbsim LANGUAGES CXX)

# ---- Compiler settings ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-O2 -Wall -Wextra)
# Align to 64 bytes and allow auto-vectorization; disable exceptions only where
# safe — we keep exceptions for the overflow guard in unit tests.
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

# ---- Executable: engine + CLI + tests ----
# Dependency-free by design (no includes outside the repo / toolchain).
# The engine classes are header-only under src/; test/engine_tests.cpp defines
# run_engine_tests() which main.cpp calls on --test.
add_executable(statarbsim
  main.cpp
  ../test/engine_tests.cpp
)
target_include_directories(statarbsim PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../test)

# ---- Microbenchmarks ----
add_executable(microbench ../bench/microbench.cpp)
target_include_directories(microbench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
