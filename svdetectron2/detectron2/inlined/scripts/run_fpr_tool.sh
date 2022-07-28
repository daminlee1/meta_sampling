#!/bin/bash
NUM_GPUS=$1
cd .. && python py2bin.py --input detectron.json --main fpr_tool --debug && cd scripts
for i in $(seq 0 $((${NUM_GPUS}-1)))
do
	CUDA_VISIBLE_DEVICES=$i python ../fpr_tool.bin --thread_id $i --threads ${NUM_GPUS} $2 &
done
