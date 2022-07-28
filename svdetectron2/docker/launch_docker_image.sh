#!/bin/bash

# Usage: launch_docker_image.sh [image name] [gpu ids]
# Example: $ ./launch_docker_image.sh root/detectron:200324-cuda9.2-cudnn7-ubuntu16.04 0,1,2,3
#     This launches a container with the name
#     'G0123.$USER.root_detectron_200324-cuda9.2-cudnn7-ubuntu16.04'
# Last updated: March 24, 2020

name=$1
name_=${name//\//_}
name_=${name_//:/_}
gpus=$2
gpus_=${gpus//,/}
port=$(( $RANDOM % 9000 + 1000 ))

nvidia-docker run -ti --rm --ipc=host --shm-size=4096m \
	--privileged \
	-v /media/hdd:/media/hdd \
	-v /data:/data \
	-v /home:/home \
	-e CUDA_VISIBLE_DEVICES=$gpus \
        -p 2$port:8080 \
        -p 3$port:8097 \
        -p 4$port:8888 \
        -p 5$port:6006 \
        -p 20990:22 \
	--name sv_alt \
	$name bash
