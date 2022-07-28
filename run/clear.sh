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
if [ -d /svdetectron2/detectron2/output/ALT.OD ];then
    rm /svdetectron2/detectron2/output/ALT.OD/*
else
    mkdir /svdetectron2/detectron2/output/ALT.OD
fi
if [ -d /svdetectron2/detectron2/output/ALT.SOD ];then
    rm /svdetectron2/detectron2/output/ALT.SOD/*
else
    mkdir /svdetectron2/detectron2/output/ALT.SOD
fi
if [ -d /svdetectron2/detectron2/output/ALT.TSTLD ];then
    rm /svdetectron2/detectron2/output/ALT.TSTLD/*
else
    mkdir /svdetectron2/detectron2/output/ALT.TSTLD
fi
if [ -d /svdetectron2/detectron2/output/ALT.OUT_IMG ];then
    rm /svdetectron2/detectron2/output/ALT.OUT_IMG/*
else
    mkdir /svdetectron2/detectron2/output/ALT.OUT_IMG
fi

# Remove previous sampling output
if [ -d /scan/cluster_output ];then
    rm /scan/cluster_output/*
    if [ -d /scan/cluster_output/cluster_show ];then
        rm /scan/cluster_output/cluster_show/*
    fi
fi

# Remove previous output images
if [ -d /output/JPEGImages ];then
    rm /output/JPEGImages/*
    rm /output/*.txt
else
    mkdir /output/JPEGImages
fi
