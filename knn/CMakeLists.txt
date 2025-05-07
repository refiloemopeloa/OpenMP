cmake_minimum_required(VERSION 3.31.6)

set(CMAKE_CXX_STANDARD 17)

project(KNN)

# Find OpenMP package
find_package(OpenMP REQUIRED)

add_executable(knn knn.cpp)

add_executable(knn-parallel knn-parallel.cpp)
# Link OpenMP to knn-parallel
target_link_libraries(knn-parallel PRIVATE OpenMP::OpenMP_CXX)

add_executable(k-nearest-neighbours k-nearest-neighbours.cpp)
# Link OpenMP to k-nearest-neighbours
target_link_libraries(k-nearest-neighbours PRIVATE OpenMP::OpenMP_CXX)