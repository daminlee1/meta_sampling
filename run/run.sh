if [ $# -ne 1 ];then
    echo error: invalid arguments.
    echo guide: run.sh '${video_list_path}'
    exit
fi

# Install prerequisite packages of ALT
pip3 install -e /svdetectron2/detectron2/

##################------------- EXTRACT FRAMES -------------##################

# Delete remaining data folders & make data folders
if [ -d /data/JPEGImages ];then
    rm /data/JPEGImages/*
else
    mkdir /data/JPEGImages
fi
if [ -d /data/Annotations ];then
    rm /data/Annotations/*
else
    mkdir /data/Annotations
fi
if [ -d /data/ImageSets ];then
    rm /data/ImageSets/*
else
    mkdir /data/ImageSets
fi

while IFS= read -r line
do
    video_list+=("$line")
done < "$1"

# Extract frames from input video
for i in ${video_list[@]}; do
    ffmpeg -i /video/$i -vf fps=1/2 /data/JPEGImages/$i%06d.png
done

# Make imageset of whole extracted frames
ls /data/JPEGImages/ &> /data/ImageSets/all.txt
sed -i 's/.png//' /data/ImageSets/all.txt

##################------------- ALT -------------##################

# Remove previous annotations
if [ -d ./output/ALT.OD ];then
    rm ./output/ALT.OD/*
else
    mkdir ./output/ALT.OD
fi
if [ -d ./output/ALT.SOD ];then
    rm ./output/ALT.SOD/*
else
    mkdir ./output/ALT.SOD
fi
if [ -d ./output/ALT.TSTLD ];then
    rm ./output/ALT.TSTLD/*
else
    mkdir ./output/ALT.TSTLD
fi

# Run ALT
cd /svdetectron2/detectron2
python3 demo/run_od_alt.py --config-file configs/Base-RCNN-BiFPN-Eff-godsod.yaml --features "od tstld sod" --input /data/JPEGImages/ --model_type ffc --output ./output --thresh-set "0.5 0.9 0.5 0.9 0.5 0.9" --save-outimg 1 --save-outtxt --opt MODEL.WEIGHTS ./release/ffc_od_tstld_sod_model.pth

# Merge ALT's annotations
python3 tools/udb_combine.py -xml1 ./output/ALT.OD -xml2 ./output/ALT.SOD -xml3 ./output/ALT.TSTLD -output /data/Annotations

##################------------- SCAN SAMPLING -------------##################

# Run SCAN(sampling)
cd /scan
python3 moco.py --config_env ./configs/env.yml --config_exp ./configs/pretext/moco_sv.yml --gpus 0
python3 scan.py --config_env ./configs/env.yml --config_exp ./configs/scan/scan_svkpi_v2.yml --gpus 0

# Remove previous sampling output
if [ -d ./cluster_output ];then
    rm ./cluster_output/*
    if [ -d ./cluster_output/cluster_show ];then
        rm ./cluster_output/cluster_show/*
    fi
fi

NUMIMAGES=$(find '/data/JPEGImages/' -type f | wc -l)
NUMSAMPLING=$(((NUMIMAGES * 2) / 3))

# Sampling data
python3 sampling_cluster.py --target_imgset /data/ImageSets/all.txt --target_data /data/ --output_dir ./cluster_output --num_cluster 27 --total_sampling ${NUMSAMPLING} --num_class 15 --show true

# Copy sampling imageset from scan directory to output directory
cp /scan/cluster_output/all_scan.txt /output/all_scan.txt

# Remove previous output images
if [ -d /output/JPEGImages ];then
    rm /output/JPEGImages/*
else
    mkdir /output/JPEGImages
fi

# Copy sampling images from data directory to output directory
while IFS= read -r line
do
    cp /data/JPEGImages/$line.png /output/JPEGImages/
done < /output/all_scan.txt

echo "done sampling"