# Compile OpenCV with imgproc for DALI Plugin
FROM ubuntu:22.04 AS opencv-build
WORKDIR /opt/opencv
RUN --mount=type=cache,target=/var/cache/apt apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y cmake g++ wget unzip ninja-build
RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/refs/tags/4.7.0.zip && \
    unzip opencv.zip   
RUN cmake -B build -S opencv-4.7.0 -G Ninja -DBUILD_LIST=imgproc && \
    cmake --build build --parallel

FROM mu00120825.eng.monash.edu.au:5000/konductor:pytorch-main

USER root
# Install OpenCV from compile container
COPY --from=opencv-build /opt/opencv /opt/opencv

# Add test toolchain for gcc-13
RUN --mount=type=cache,target=/var/cache/apt apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y software-properties-common && \
    add-apt-repository ppa:ubuntu-toolchain-r/test 

# Install build tools
RUN --mount=type=cache,target=/var/cache/apt apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    ninja-build cmake libtbb-dev g++-13 gcc-13

RUN cmake --install /opt/opencv/build && rm -r /opt/opencv

# Install in dist-utils so not overwritten in /home/worker
RUN pip3 install \
    opencv-python-headless==4.7.0.68 \
    einops==0.6.0 \
    scipy==1.10.0 \ 
    git+https://github.com/rtqichen/torchdiffeq.git@7265eb764e97cc485ec2d8fcbd87b4b95ca416e8

# COMMIT arg is Required for training tracking purposes
# Use --build-arg COMMIT="$(git rev-parse --short HEAD)"
ARG COMMIT
RUN [ ! -z "${COMMIT}" ]
ENV COMMIT_SHA=${COMMIT}

USER worker
WORKDIR /home/worker

COPY --chown=worker:worker . .
RUN cd src/dataset/plugins && \
    CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13 \
    cmake -B build -G Ninja && \
    cmake --build build --parallel --config Release
