FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
RUN apt-get update --fix-missing && \
    apt-get install -y python-dev python-pip libglib2.0-0 libsm6 libxrender1 libxext6 && \
    pip install --upgrade pip && \
    python -m pip install future opencv-python pyyaml==3.12 six protobuf pycrypto cython && \
    python -m pip install pycocotools
RUN apt-get install -y software-properties-common && \
    add-apt-repository ppa:jonathonf/ffmpeg-3 -y && \
    apt-get update --fix-missing && \
    apt-get install -y pkg-config libavformat-dev libavcodec-dev libswscale-dev libavdevice-dev ffmpeg && \
    python -m pip install av pillow
RUN apt-get install -y libatlas-base-dev libopencv-dev && \
    apt-get install -y --no-install-recommends libboost-all-dev
ADD so_libs.tar /usr/local/lib/
ADD py_libs.tar /usr/local/lib/python2.7/dist-packages/
ADD dist.tar /opt/
