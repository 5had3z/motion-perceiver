FROM nvcr.io/nvidia/tensorflow:23.06-tf2-py3

RUN apt-get update && apt-get install -y libopenexr-dev
RUN pip3 install \
    waymo_open_dataset_tf_2_11_0==1.5.2 \
    typer==0.9.0 \
    konductor==0.0.3
