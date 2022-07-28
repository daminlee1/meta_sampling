# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import numpy as np
import os
import xml.etree.ElementTree as ET
from fvcore.common.file_io import PathManager

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

import logging
import tarfile
from detectron2.data.datasets import udb_datasets 
# from joblib import parallel, delayed
# import multiprocessing
# import parmap


logger = logging.getLogger(__name__)


__all__ = ["register_udb", "register_3dcar_udb"]


# fmt: off
CLASS_NAMES = [
    "pedestrian", "rider", "car", "truck", "bus", "ts_circle", "ts_triangle", "ts_rectangle", "tl", "ignored",
]

CLASS_NAMES_3D = [
    "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9",
]
# fmt: on


def load_udb_instances(udb_tarfile: str):
    """
    Load Pascal VOC detection annotations to Detectron2 format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
    """
    # tar = tarfile.open(udb_tarfile)
    tar = tarfile.open(udb_tarfile)
    

    ann_file_list = [name for name in tar.getnames() if 'Annotations/' in name]
    dicts = []

    cls_dict = udb_datasets.get_god_2020_10cls_dataset()            

    for ann_file in ann_file_list:
        f = tar.extractfile(tar.getmember(ann_file))        
        tree = ET.parse(f)
        
        # jpeg_file = ann_file.replace("Annotations", "JPEGImages").replace("xml", "jpg")
        jpeg_file = os.path.join(udb_tarfile, ann_file.replace("Annotations", "JPEGImages").replace("xml", "jpg"))
        fileid = ann_file_list.index(ann_file)

        r = {
            "file_name": jpeg_file,
            "image_id": fileid,
            "height": int(tree.findall("./size/height")[0].text),
            "width": int(tree.findall("./size/width")[0].text),
        }
        instances = []

        for obj in tree.findall("object"):
            cls = obj.find("name").text
            # We include "difficult" samples in training.
            # Based on limited experiments, they don't hurt accuracy.
            # difficult = int(obj.find("difficult").text)
            # if difficult == 1:
            # continue
            bbox = obj.find("bndbox")
            bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            # Original annotations are integers in the range [1, W or H]
            # Assuming they mean 1-based pixel indices (inclusive),
            # a box with annotation (xmin=1, xmax=W) covers the whole image.
            # In coordinate space this is represented by (xmin=0, xmax=W)
            bbox[0] -= 1.0
            bbox[1] -= 1.0
            instances.append(
                {"category_id": cls_dict.cls_map[cls] - 1, "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS}
            )
        r["annotations"] = instances
        dicts.append(r)

    logger.info("Loaded {} images in UDB format from {}".format(len(ann_file_list), udb_tarfile))

    return dicts



def register_udb(name, tarfile):
    DatasetCatalog.register(name, lambda: load_udb_instances(tarfile))
    MetadataCatalog.get(name).set(
        thing_classes=CLASS_NAMES, tarfile=tarfile
    )



##################################


def process_xml(udb_tarfile, tarobj, ann_file, ann_file_list):
    if os.path.isdir(udb_tarfile):
        with open(ann_file) as f:
            tree = ET.parse(f)
        jpeg_file = ann_file.replace("Annotations", "JPEGImages").replace("xml", "jpg")    
    else:
        f = tarobj.extractfile(tarobj.getmember(ann_file))
        tree = ET.parse(f)
        jpeg_file = os.path.join(udb_tarfile, ann_file.replace("Annotations", "JPEGImages").replace("xml", "jpg"))
    

    fileid = ann_file_list.index(ann_file)

    filexml = tree.getroot()
    imgwidth = filexml.attrib["ImageWidth"]
    imgheight = filexml.attrib["ImageHeight"]

    r = {
        "file_name": jpeg_file,
        "image_id": fileid,
        "height": np.int(imgheight),
        "width": np.int(imgwidth),
    }
    instances = []

    car3ds = tree.findall("Car3D")
    for car3d in car3ds:
        direction = car3d.attrib["Direction"]
        shape = car3d.attrib["Shape"]
        point = []
        for p in car3d.iter("Point"):
            # point.append([p.attrib["x"], p.attrib["y"]])
            point = point + [np.float(p.attrib["x"]), np.float(p.attrib["y"])]

        instances.append(
            # {"point": point, "direction": direction, "shape": shape}
            {"point": point, "num_pts": len(point)/2, "direction": np.int(direction)-1, "shape": np.int(shape)}
        )
    r["annotations"] = instances
    return r


def load_udb_3d_instances(udb_tarfile: str):
    """
    Load Pascal VOC detection annotations to Detectron2 format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
    """

    imagesets = ['harman_train.txt', 'lge_train.txt', 'movon_ger_train.txt']

    if os.path.isdir(udb_tarfile):
        imset_files = [os.path.join(udb_tarfile, 'ImageSets/' + name) for name in imagesets]
        tar = None
    else:
        tar = tarfile.open(udb_tarfile)    
        # imagesets = ['harman_train.txt', 'lge_train.txt', 'movon_ger_train.txt']
        imagesets = ['harman_train.txt']
        imset_files = [tar.getmember('ImageSets/' + name) for name in imagesets]

    ann_file_list = []

    for imfile in imset_files:
        if os.path.isdir(udb_tarfile):
            with open(imfile, "r") as file:
                imfilenames = file.read().splitlines()
            annfilenames = [os.path.join(udb_tarfile, 'Annotations/' + imfile + '.xml') for imfile in imfilenames]            
        else:
            f = tar.extractfile(imfile)
            imfilenames = f.read().splitlines()
            annfilenames = ['Annotations/' + imfile.decode('ascii') + '.xml' for imfile in imfilenames]
        # annfilenames = annfilenames[1:20]
        ann_file_list = ann_file_list + annfilenames

    dicts = []

    # num_cores = multiprocessing.cpu_count()
    # split_ann_list = np.array_split(ann_file_list, num_cores)
    # split_ann_list = [x.tolist() for x in split_ann_list]

    # result = parmap.map(process_xml, tar, split_ann_list, ann_file_list, pm_pbar=True, pm_processes=num_cores)

    for ann_file in ann_file_list:
        r = process_xml(udb_tarfile, tar, ann_file, ann_file_list)
        dicts.append(r)
        # print('[{}/{}]'.format(ann_file_list.index(ann_file), len(ann_file_list)))

    logger.info("Loaded {} images in UDB format from {}".format(len(ann_file_list), udb_tarfile))

    return dicts



def register_3dcar_udb(name, tarfile):
    DatasetCatalog.register(name, lambda: load_udb_3d_instances(tarfile))
    MetadataCatalog.get(name).set(
        thing_classes=CLASS_NAMES_3D, tarfile=tarfile
    )