FROM nvidia/cuda:11.0-cudnn8-devel-ubuntu18.04

RUN rm /etc/apt/sources.list.d/cuda.list
#RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get clean
RUN apt update
ENV DEBIAN_FRONTEND=noninteractive
RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt install python3.9 -y
RUN apt list *distutils*
RUN apt-get install -y python3.9-distutils -y
RUN apt install curl -y
RUN apt-get install git -y
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.9 get-pip.py --force-reinstall
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 2
RUN update-alternatives --config python3
# RUN git clone https://daminlee1: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@github.com/StradVision/scan.git
# WORKDIR "./scan"
RUN pip3 install numpy
RUN pip3 install faiss-gpu
RUN pip3 install opencv-python
RUN apt-get install libgl1-mesa-glx -y
RUN pip3 install matplotlib
RUN pip3 install seaborn
RUN pip3 install tqdm
RUN pip3 install torch
RUN pip3 install torchvision
RUN pip3 install torchsummary
RUN pip3 install pynvml
RUN pip3 install pyyaml
RUN pip3 install easydict
RUN pip3 install termcolor
RUN pip3 install sklearn
RUN pip3 install tensorboardX
RUN apt-get install wget -y
RUN apt-get install ffmpeg -y

RUN apt update -y
RUN apt upgrade -y
RUN apt install python2.7 -y
RUN apt install python-pip -y
RUN alias python=python2.7

#svcaffe
RUN apt update && apt install libpython-dev python python-pip cmake gcc vim libopencv-dev libopenblas-dev libhdf5-dev libatlas-base-dev build-essential libboost-all-dev libprotobuf-dev libgoogle-glog-dev libgflags-dev libhdf5-serial-dev liblmdb-dev libleveldb-dev libsnappy-dev protobuf-compiler -y
RUN pip2 install numpy cython
RUN python -m pip install --upgrade pip
RUN python -m pip install wheel opencv-python==3.2.0.8 cython scikit-image protobuf

#add local host path to docker container
# ADD ./video /video
# ADD ./output /output
# ADD ./data /data

#svdetectron2
RUN apt install libpython3.9-dev -y
RUN pip3 install fvcore
RUN pip3 install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install pycocotools
# RUN apt-get update --fix-missing
# RUN apt-get install -y wget
# RUN wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
# RUN bash Anaconda3-2020.02-Linux-x86_64.sh -b -p /opt/conda && \
#     rm Anaconda3-2020.02-Linux-x86_64.sh
# RUN /opt/conda/bin/conda create -n svdetectron2 python=3.7
# RUN export CMAKE_PREFIX_PATH="/opt/conda"
# # pytorch
# RUN /bin/bash -c "source /opt/conda/bin/activate svdetectron2 && conda install -y git numpy ninja pyyaml mkl mkl-include setuptools cmake cffi Cython cloudpickle tensorboard"
# RUN /bin/bash -c "source /opt/conda/bin/activate svdetectron2 && conda install -c pytorch magma-cuda101"
# # detectron2
# RUN /bin/bash -c "source /opt/conda/bin/activate svdetectron2 && conda install -c conda-forge fvcore"
# RUN /bin/bash -c "source /opt/conda/bin/activate svdetectron2 && cd ~ && git clone https://github.com/cocodataset/cocoapi.git && cd cocoapi/PythonAPI && make install"
# RUN sed -i '$a export PATH=/opt/conda/bin:$PATH' ~/.bashrc && \
#     sed -i '$a export LD_LIBRARY_PATH=/opt/pytorch/build/lib:$LD_LIBRARY_PATH' ~/.bashrc && \
#     sed -i '$a export PYTHONPATH=/opt/pytorch/build:$PYTHONPATH' ~/.bashrc && \
#     sed -i '$a source /opt/conda/bin/activate svdetectron2' ~/.bashrc