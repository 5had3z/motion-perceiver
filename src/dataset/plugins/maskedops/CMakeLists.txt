cmake_minimum_required(VERSION 3.21.3 FATAL_ERROR)

add_library(maskedops SHARED maskedops.cpp)
target_compile_features(maskedops PRIVATE cxx_std_17 c_std_11)

set_target_properties(maskedops
    PROPERTIES
    CUDA_STANDARD "14"
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "86;80;75;70;61"
)

target_compile_options(
    maskedops
    PRIVATE
    $<$<CONFIG:Debug>:-O0;-g;-march=native;${WARNING_OPTIONS}>
    $<$<CONFIG:Release>:-O3;-march=native;${WARNING_OPTIONS}>
)

target_compile_definitions(maskedops PRIVATE ${DALI_DEFINE_FLAGS})

target_link_directories(maskedops PRIVATE ${DALI_LIB_DIR})

target_include_directories(maskedops PRIVATE ${CUDAToolkit_INCLUDE_DIRS} ${DALI_INCLUDE_DIR})

target_link_libraries(maskedops dali)
