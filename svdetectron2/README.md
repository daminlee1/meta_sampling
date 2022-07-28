# svdetectron2
detectron2 customized by StradVision for developing DL algorithms

# Installation

### Clone this repository
```
 $ git clone https://github.com/StradVision/svdetectron2.git
```

### Download pretrained weights
```
/working_directory$ cd svdetectron2/detectron2
/working_directory/svdetectron2/detectron2$ mkdir weights
/working_directory/svdetectron2/detectron2$ cd weights
/working_directory/svdetectron2/detectron2/weights$ sftp [COCO ID]@10.50.20.16
sftp> mget /shared/Dataset/GOD/sv_god_data/weights/ALT/*.pkl
sftp> exit
```

### Build svdetectron2 docker image
```
/working_directory/svdetectron2/docker$ sudo ./build_dockerfile.sh svdetectron2:develop develop
```
- Prerequisites: NVIDIA driver, docker, nvidia-docker2
- Refer to the "install_nvidia_docker.sh" for installing nvidia-docker2

### Run svdetectron2 docker image
```
/working_directory/svdetectron2/docker$ sudo ./launch_docker_image.sh root/svdetectron2:develop 0
```
- Modify "launch_docker_image.sh" file to access the working directory in the docker container
- The following directories has been mounted by default: /media/hdd, /data, /home

### Build PyTorch/Caffe2 and torchvision
```
# build pytorch & caffe2
(svdetectron2) /workding_directory$ git clone --recursive https://github.com/pytorch/pytorch && cd pytorch
(svdetectron2) /workding_directory/pytorch$ git submodule sync
(svdetectron2) /workding_directory/pytorch$ git submodule update --init --recursive
(svdetectron2) /workding_directory/pytorch$ cp ../svdetectron2/pytorch/caffe2/operators/dirichlet_loss_op.* caffe2/operators
(svdetectron2) /workding_directory/pytorch$ export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
(svdetectron2) /workding_directory/pytorch$ python setup.py develop
(svdetectron2) /workding_directory/pytorch$ ln -s `pwd` /opt/pytorch

# build torchvision
(svdetectron2) /workding_directory/pytorch$ cd ..
(svdetectron2) /workding_directory$ git clone https://github.com/pytorch/vision.git
(svdetectron2) /workding_directory$ cd vision
(svdetectron2) /workding_directory/vision$ python setup.py install
```

### Build svdetectron2
```
(svdetectron2) /workding_directory/pytorch$ cd ../svdetectron2
(svdetectron2) /workding_directory/svdetectron2$ python -m pip install -e detectron2
```




