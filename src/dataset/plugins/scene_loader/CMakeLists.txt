cmake_minimum_required(VERSION 3.21.3 FATAL_ERROR)

find_package(OpenCV REQUIRED COMPONENTS imgproc)

CPMAddPackage(
    NAME yaml-cpp
    GIT_TAG 0.8.0
    GITHUB_REPOSITORY jbeder/yaml-cpp
)

add_library(scene_loader SHARED src.cpp)
target_compile_features(scene_loader PRIVATE cxx_std_17 c_std_11)

target_compile_options(
    scene_loader
    PRIVATE
    $<$<CONFIG:Debug>:-O0;-g;-march=native;${WARNING_OPTIONS}>
    $<$<CONFIG:Release>:-O3;-march=native;${WARNING_OPTIONS}>
)

target_compile_definitions(scene_loader PRIVATE ${DALI_DEFINE_FLAGS})

target_link_directories(scene_loader PRIVATE ${DALI_LIB_DIR})

target_include_directories(scene_loader PRIVATE ${CUDAToolkit_INCLUDE_DIRS} ${DALI_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})

target_link_libraries(scene_loader dali yaml-cpp ${OpenCV_LIBS})
