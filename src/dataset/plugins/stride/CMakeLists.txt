cmake_minimum_required(VERSION 3.21.3 FATAL_ERROR)

add_library(stride SHARED plugin.cpp)
target_compile_features(stride PRIVATE cxx_std_17 c_std_11)

target_compile_options(
  stride PRIVATE $<$<CONFIG:Debug>:-O0;-g;-march=native;${WARNING_OPTIONS}>
                 $<$<CONFIG:Release>:-O3;-march=native;${WARNING_OPTIONS}>)

target_link_directories(stride PRIVATE ${DALI_LIB_DIR})

target_include_directories(
  stride PRIVATE ${CUDAToolkit_INCLUDE_DIRS} ${DALI_INCLUDE_DIR}
                 ${OpenCV_INCLUDE_DIRS})

target_link_libraries(stride dali)
