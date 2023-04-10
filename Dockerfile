FROM WITHHELD/konductor:pytorch-main

# Install opencv 4.5 for DALI
USER root
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y libopencv-dev

# USER worker
RUN pip3 install \
    opencv-python-headless \
    einops==0.6.0 \
    scipy==1.10.0

COPY . /home/worker
WORKDIR /home/worker
RUN cd src/dataset/plugins && sh makeplugins.sh

ENTRYPOINT [ "torchrun" ]
