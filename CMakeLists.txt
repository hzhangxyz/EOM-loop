cmake_minimum_required(VERSION 3.13)
project(EOM-loop)

add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")

find_package(pybind11 QUIET)
if(pybind11_FOUND)
   message("-- Enable python(system)")
else()
   add_subdirectory(pybind11)
   message("-- Enable python(local)")
endif()

pybind11_add_module(matrix_U matrix_U.cpp)
pybind11_add_module(matrix_A matrix_A.cpp)
