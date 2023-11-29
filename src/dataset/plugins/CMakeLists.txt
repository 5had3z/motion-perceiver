cmake_minimum_required(VERSION 3.21.3 FATAL_ERROR)

project(dali_plugins LANGUAGES CXX C)

find_package(CUDAToolkit REQUIRED COMPONENTS cudart)

set(CMAKE_CXX_STANDARD_REQUIRED ON)

execute_process(
  COMMAND python3 -c
          "import nvidia.dali as dali; print(dali.sysconfig.get_lib_dir())"
  OUTPUT_VARIABLE DALI_LIB_DIR)
string(STRIP ${DALI_LIB_DIR} DALI_LIB_DIR)

execute_process(
  COMMAND
    python3 -c
    "import nvidia.dali as dali; print(\" \".join(dali.sysconfig.get_compile_flags()))"
  OUTPUT_VARIABLE DALI_COMPILE_FLAGS)
string(STRIP ${DALI_COMPILE_FLAGS} DALI_COMPILE_FLAGS)

# Extract define flags "-D.." from the compile flags, this is usually
# -D_GLIBCXX_USE_CXX11_ABI=0
string(REGEX MATCH "-D[A-Za-z0-9_=]+" DALI_DEFINE_FLAGS ${DALI_COMPILE_FLAGS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${DALI_DEFINE_FLAGS}")

execute_process(
  COMMAND python3 -c
          "import nvidia.dali as dali; print(dali.sysconfig.get_include_dir())"
  OUTPUT_VARIABLE DALI_INCLUDE_DIR)
string(STRIP ${DALI_INCLUDE_DIR} DALI_INCLUDE_DIR)

set(WARNING_OPTIONS "-Wall;-Wfatal-errors;-Wextra;-Wshadow;-Wpedantic")

include(cmake/CPM.cmake)

add_subdirectory(maskedops)

add_subdirectory(maskfrompoly)

add_subdirectory(roadgraph)

add_subdirectory(random)

add_subdirectory(stride)

add_subdirectory(scene_loader)
