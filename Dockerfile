# Compile OpenCV with imgproc for DALI Plugin
FROM ubuntu:22.04 AS opencv-build
WORKDIR /opt/opencv
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y cmake g++ wget unzip ninja-build
RUN wget https://github.com/opencv/opencv/archive/refs/tags/4.10.0.tar.gz && \
    tar -xzf 4.10.0.tar.gz
RUN cmake -B build -S opencv-4.10.0 -G Ninja -DBUILD_LIST=imgproc && \
    cmake --build build --parallel --config Release

# Main image build
FROM nvcr.io/nvidia/pytorch:24.08-py3 AS main-build

# Add test toolchain for gcc-13
RUN --mount=type=cache,target=/var/cache/apt apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y software-properties-common && \
    add-apt-repository ppa:ubuntu-toolchain-r/test

# Install build tools
RUN --mount=type=cache,target=/var/cache/apt apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    ninja-build cmake libtbb-dev g++-13 gcc-13

# Add code cli for remote tunnel
RUN curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' --output vscode_cli.tar.gz && \
    tar -xf vscode_cli.tar.gz -C /usr/local/bin && \
    rm vscode_cli.tar.gz

# Install OpenCV from compile container
COPY --from=opencv-build /opt/opencv /opt/opencv
RUN cmake --install /opt/opencv/build && rm -r /opt/opencv

# Install in dist-utils so not overwritten in /home/worker
RUN pip3 install \
    einops==0.8.0 \
    paramiko==3.4.1 \
    git+https://github.com/rtqichen/torchdiffeq.git@7265eb764e97cc485ec2d8fcbd87b4b95ca416e8 \ 
    git+https://github.com/5had3z/konductor.git@61303d5e854a779ed214a8a9ac35cddd7e5f80cf

RUN useradd -rm -d /home/worker -s /bin/bash -G sudo -U -u 1000 worker
USER worker
WORKDIR /home/worker

# COMMIT arg is Required for training tracking purposes
# Use --build-arg COMMIT="$(git rev-parse --short HEAD)"
ARG COMMIT
RUN [ ! -z "${COMMIT}" ]
ENV COMMIT_SHA=${COMMIT}

COPY --chown=worker:worker . .
RUN cd src/dataset/plugins && \
    CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13 \
    cmake -B build -G Ninja && \
    cmake --build build --parallel --config Release
