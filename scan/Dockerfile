FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
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
RUN git clone https://daminlee1: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@github.com/StradVision/scan.git
WORKDIR "./scan"
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
RUN apt-get install wget -y
