FROM ubuntu:22.04 AS opencv-build
WORKDIR /opt/opencv
RUN --mount=type=cache,target=/var/cache/apt apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y cmake g++ wget unzip
RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/refs/tags/4.7.0.zip && \
    unzip opencv.zip && \
    cmake -B build -S opencv-4.7.0 -DBUILD_LIST=imgproc
RUN cd build && make -j$(nproc)

FROM WITHHELD/konductor:pytorch-main

# Install opencv 4.5 for DALI, only install imgproc
USER root
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y make
COPY --from=opencv-build /opt/opencv /opt/opencv
RUN cd /opt/opencv/build && make install && cd /root && rm -r /opt/opencv

RUN pip3 install \
    opencv-python-headless \
    einops==0.6.0 \
    scipy==1.10.0

# COMMIT arg is Required for training tracking purposes
# Use --build-arg COMMIT="$(git rev-parse --short HEAD)"
ARG COMMIT
RUN [ ! -z "${COMMIT}" ]
ENV COMMIT_SHA=${COMMIT}

COPY --chown=worker:worker . /home/worker
WORKDIR /home/worker
RUN cd src/dataset/plugins && sh makeplugins.sh
USER worker
