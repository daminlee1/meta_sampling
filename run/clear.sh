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

# Remove previous sampling output
if [ -d ./cluster_output ];then
    rm ./cluster_output/*
    if [ -d ./cluster_output/cluster_show ];then
        rm ./cluster_output/cluster_show/*
    fi
fi

# Remove previous output images
if [ -d /output/JPEGImages ];then
    rm /output/JPEGImages/*
    rm /output/*.txt
else
    mkdir /output/JPEGImages
fi
