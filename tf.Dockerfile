# Tensorflow dockerfile for testing things and running results export
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# Waymo will download tensorflow as a dependency
USER root
RUN apt-get install -y libopenexr-dev && \
    pip3 install pytest waymo_open_dataset_tf_2_11_0==1.6.1

USER worker
