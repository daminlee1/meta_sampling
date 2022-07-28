#!/bin/bash
NUM_GPUS=$1
cd .. && python py2bin.py --input detectron.json --main mask_worker --debug && cd scripts
for i in $(seq 0 $((${NUM_GPUS}-1)))
do
	CUDA_VISIBLE_DEVICES=$i ../launcher ../mask_worker.bin &
done
