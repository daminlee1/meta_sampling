#!/bin/bash

# Usage: build_dockerfile.sh [image name] [directory containing Dockerfile]
# Example: $ ./build_dockerfile.sh detectron:180619-cuda9.0-cudnn7-ubuntu16.04 detectron_180619
#     This builds an image with the name
#     '$USER/detectron:180619-cuda9.0-cudnn7-ubuntu16.04'
# Last updated: June 19, 2018

owner=$USER
name=$1
path=$2

nvidia-docker build -t $owner/$name $path
