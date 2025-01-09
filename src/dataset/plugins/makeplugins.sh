# Set gcc/g++ 13 as compilers for C++23 support
CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13 cmake -B build -G Ninja
cmake --build build --parallel --config Release
