FROM nvcr.io/nvidia/tensorflow:23.06-tf2-py3

RUN apt-get update && apt-get install -y libopenexr-dev
RUN pip3 install \
    waymo_open_dataset_tf_2_11_0==1.5.2 \
    git+https://github.com/5had3z/konductor.git
