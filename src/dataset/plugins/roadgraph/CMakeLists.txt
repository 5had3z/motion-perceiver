cmake_minimum_required(VERSION 3.21.3 FATAL_ERROR)

find_package(OpenCV REQUIRED COMPONENTS imgproc)

add_library(roadgraph SHARED roadgraph.cpp)
target_compile_features(roadgraph PRIVATE cxx_std_17 c_std_11)

target_compile_options(
  roadgraph PRIVATE $<$<CONFIG:Debug>:-O0;-g;-march=native;${WARNING_OPTIONS}>
                    $<$<CONFIG:Release>:-O3;-march=native;${WARNING_OPTIONS}>)

target_link_directories(roadgraph PRIVATE ${DALI_LIB_DIR})

target_include_directories(
  roadgraph SYSTEM PRIVATE ${CUDAToolkit_INCLUDE_DIRS} ${DALI_INCLUDE_DIR}
                           ${OpenCV_INCLUDE_DIRS})

target_link_libraries(roadgraph dali ${OpenCV_LIBS})
