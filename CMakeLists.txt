cmake_minimum_required(VERSION 2.8)
project(new_cuboid_method_cpp)

set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )
set( CMAKE_BUILD_TYPE Release)


find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(include)
include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/third_party/tictoc_profiler/include
  ${PROJECT_SOURCE_DIR}/third_party/line_lbd/include
)

## Declare a C++ library
add_library(filter_cube  SHARED
src/filter_cube.cpp )

target_link_libraries(filter_cube   ${OpenCV_LIBRARIES} ${EIGEN3_LIBRARIES} 
${PROJECT_SOURCE_DIR}/third_party/line_lbd/libline_lbd_lib.so
${PROJECT_SOURCE_DIR}/third_party/tictoc_profiler/libtictoc_profiler.so
)


add_executable(${PROJECT_NAME} src/main.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES} ${EIGEN3_LIBRARIES} filter_cube
${PROJECT_SOURCE_DIR}/third_party/line_lbd/libline_lbd_lib.so)
