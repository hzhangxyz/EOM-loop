cmake_minimum_required(VERSION 3.13)
project(EOM-loop)

find_package(pybind11 REQUIRED)

pybind11_add_module(matrix_U matrix_U.cpp)
pybind11_add_module(matrix_A matrix_A.cpp)
