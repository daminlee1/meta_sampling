echo $1
if [ $# -ne 1 ];then
    echo error: invalid arguments.
    echo guide: start.sh '${gpu_info}'
    echo 'gpu_info: 0(titanRTX), 1(RTX3090)'
    exit
fi

if [ $1 == 0 ];then
    cd ./docker/titanrtx
    docker build --tag meta_sampling_cuda11.0_cudnn8:0.0 .
    cd ../../
    docker run -it --rm --gpus all --shm-size 16G -v  ${PWD}/data:/data -v ${PWD}/video:/video -v ${PWD}/output:/output -v ${PWD}/svdetectron2:/svdetectron2 -v ${PWD}/run:/run -v ${PWD}/scan:/scan --name meta meta_sampling_cuda11.0_cudnn8:0.0 /bin/bash
elif [ $1 == 1 ];then
    cd ./docker/rtx3090
    docker build --tag meta_sampling_cuda11.2_cudnn8:0.0 .
    cd ../../
    docker run -it --rm --gpus all --shm-size 16G -v  ${PWD}/data:/data -v ${PWD}/video:/video -v ${PWD}/output:/output -v ${PWD}/svdetectron2:/svdetectron2 -v ${PWD}/run:/run -v ${PWD}/scan:/scan --name meta meta_sampling_cuda11.2_cudnn8:0.0 /bin/bash
fi
