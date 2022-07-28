FROM nvidia/cuda:11.2.2-devel-ubuntu18.04
RUN apt-get update --fix-missing
RUN apt-get install -y wget
RUN wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
RUN bash Anaconda3-2020.02-Linux-x86_64.sh -b -p /opt/conda && \
    rm Anaconda3-2020.02-Linux-x86_64.sh
RUN /opt/conda/bin/conda create -n svdetectron2 python=3.7
RUN export CMAKE_PREFIX_PATH="/opt/conda"
# pytorch
RUN /bin/bash -c "source /opt/conda/bin/activate svdetectron2 && conda install -y git numpy ninja pyyaml mkl mkl-include setuptools cmake cffi Cython cloudpickle tensorboard"
RUN /bin/bash -c "source /opt/conda/bin/activate svdetectron2 && conda install -c pytorch magma-cuda101"
# detectron2
RUN /bin/bash -c "source /opt/conda/bin/activate svdetectron2 && conda install -c conda-forge fvcore"
RUN /bin/bash -c "source /opt/conda/bin/activate svdetectron2 && cd ~ && git clone https://github.com/cocodataset/cocoapi.git && cd cocoapi/PythonAPI && make install"
RUN sed -i '$a export PATH=/opt/conda/bin:$PATH' ~/.bashrc && \
    sed -i '$a export LD_LIBRARY_PATH=/opt/pytorch/build/lib:$LD_LIBRARY_PATH' ~/.bashrc && \
    sed -i '$a export PYTHONPATH=/opt/pytorch/build:$PYTHONPATH' ~/.bashrc && \
    sed -i '$a source /opt/conda/bin/activate svdetectron2' ~/.bashrc

RUN pip install opencv-python
RUN apt-get install -y libgl1-mesa-glx
RUN pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html

