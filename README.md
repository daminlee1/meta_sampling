# Meta sampling

## Introduction
***
Sampling the data from video according to each data’s meta-info, and it is called **“Meta sampling”**. Meta sampling is clustering data and sampling data from each cluster, so we can sampling more valuable data and balanced distribution. (less data redundancy)

## Prerequisite
***
OS : Ubuntu or Centos (No Windows)

GPU : NVIDIA TITAN RTX / NVIDIA Geforce RTX 3090

NVIDIA Driver >= 450.80 

Docker >= 19.03

## How to use
***


```bash
# Build docker image and run docker container
./start.sh ${GPU_INFO}
```

- GPU_INFO

  input your gpu model

  0 : TitanRTX

  1 : RTX3090


```bash
# Make video_list (input videos)
cd /video
./make_video_list.sh

# Run sampling
cd /run
./run.sh /video/video_list.txt
```

## Output
***

```bash
# Sampling result path
/output/
       L all_scan.txt
       L JPEGImages/
                   L img000000.png
                   L img000001.png
                   L ...
```