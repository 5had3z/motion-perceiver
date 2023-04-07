FROM nvcr.io/nvidia/tensorflow:21.11-tf2-py3

RUN apt-get update && apt-get install -y libopenexr-dev
RUN pip3 install waymo_open_dataset_tf_2_6_0==1.4.5 tensorflow_graphics==2021.12.3
