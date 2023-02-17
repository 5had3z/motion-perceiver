cmake_minimum_required(VERSION 3.21.3 FATAL_ERROR)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS imgproc)

add_library(roadgraph SHARED roadgraph.cpp)
target_compile_features(roadgraph PRIVATE cxx_std_17 c_std_11)

set_target_properties(roadgraph
    PROPERTIES
    CUDA_STANDARD "14"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86;80;75;70;61"
)

target_compile_options(
    roadgraph
    PRIVATE
    $<$<CONFIG:Debug>:-O0;-g;-march=native;${WARNING_OPTIONS}>
    $<$<CONFIG:Release>:-O3;-march=native;${WARNING_OPTIONS}>
)

target_compile_definitions(roadgraph PRIVATE ${DALI_DEFINE_FLAGS})

target_link_directories(roadgraph PRIVATE ${DALI_LIB_DIR})

target_include_directories(roadgraph PRIVATE ${CUDAToolkit_INCLUDE_DIRS} ${DALI_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})

target_link_libraries(roadgraph dali Eigen3::Eigen ${OpenCV_LIBS})
