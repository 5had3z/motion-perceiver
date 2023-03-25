FROM WITHHELD/konductor:pytorch-main

# Install opencv 4.5 for DALI
USER root
WORKDIR /opt/opencv
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y cmake g++ wget unzip
RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/refs/tags/4.7.0.zip && \
    unzip opencv.zip && \
    cmake -B build -S opencv-4.7.0
RUN cd build && make install -j12
WORKDIR /root
RUN rm -r /opt/opencv

# USER worker
RUN pip3 install \
    opencv-python-headless \
    einops==0.6.0 \
    scipy==1.10.0

COPY . /home/worker
WORKDIR /home/worker
RUN cd src/dataset/plugins && sh makeplugins.sh

ENTRYPOINT [ "torchrun" ]
