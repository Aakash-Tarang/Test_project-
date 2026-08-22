cmake_minimum_required(VERSION 3.16)
project(statarbsim LANGUAGES CXX)

# ---- Compiler settings ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-O2 -Wall -Wextra)

# ---- Executable ----
# NOTE: dependency-free by design (no includes outside the repo/toolchain).
# Part 3+ adds the real classes under src/{data,model,signal,portfolio,engine}.
add_executable(statarbsim main.cpp)

# ---- Tests (from Part 3) ----
# A --test flag in main() runs the hand-rolled unit-test harness
# (see test/minimal_test.hpp, added in Part 3).
