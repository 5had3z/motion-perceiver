cmake_minimum_required(VERSION 3.21.3 FATAL_ERROR)

find_package(OpenCV REQUIRED COMPONENTS imgproc)

add_library(maskfrompoly SHARED maskfrompoly.cpp)
target_compile_features(maskfrompoly PRIVATE cxx_std_17 c_std_11)

target_compile_options(
  maskfrompoly
  PRIVATE $<$<CONFIG:Debug>:-O0;-g;-march=native;${WARNING_OPTIONS}>
          $<$<CONFIG:Release>:-O3;-march=native;${WARNING_OPTIONS}>)

target_link_directories(maskfrompoly PRIVATE ${DALI_LIB_DIR})

target_include_directories(
  maskfrompoly PRIVATE ${CUDAToolkit_INCLUDE_DIRS} ${DALI_INCLUDE_DIR}
                       ${OpenCV_INCLUDE_DIRS})

target_link_libraries(maskfrompoly dali ${OpenCV_LIBS})
