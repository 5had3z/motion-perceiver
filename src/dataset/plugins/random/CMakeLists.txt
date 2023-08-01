cmake_minimum_required(VERSION 3.21.3 FATAL_ERROR)

add_library(dali_random SHARED plugin.cpp)
target_compile_features(dali_random PRIVATE cxx_std_23 c_std_11)

target_compile_options(
    dali_random
    PRIVATE
    $<$<CONFIG:Debug>:-O0;-g;-march=native;${WARNING_OPTIONS}>
    $<$<CONFIG:Release>:-O3;-march=native;${WARNING_OPTIONS}>
)

target_compile_definitions(dali_random PRIVATE ${DALI_DEFINE_FLAGS})

target_link_directories(dali_random PRIVATE ${DALI_LIB_DIR})

target_include_directories(dali_random PRIVATE ${CUDAToolkit_INCLUDE_DIRS} ${DALI_INCLUDE_DIR})

target_link_libraries(dali_random dali)
