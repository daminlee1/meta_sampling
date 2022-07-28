"""Provide stub objects that can act as stand-in "dummy" datasets for simple use
cases, like getting all classes in a dataset. This exists so that demos can be
run without requiring users to download/install datasets first.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from detectron2.utils.collections import AttrDict
import tarfile, os
import numpy as np
import cv2
import xml.etree.ElementTree as ET


def get_coco_dataset(udb_path=None, udb_root=None):
    """A dummy COCO dataset that includes only the 'classes' field."""
    ds = AttrDict()
    classes = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
        'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    ds.classes = {i: name for i, name in enumerate(classes)}
    ds.cls_map = { name: i for i, name in enumerate(classes) if name != '__background__' }
    ds.all_tar_files, ds.all_image_files, ds.all_xml_files, ds.all_db_names = preprocess_udb(udb_path, udb_root)
    return ds


def get_cityscapes_dataset(udb_path=None, udb_root=None):
    ds = AttrDict()
    classes = [
        '__background__', 'bicycle', 'car', 'traffic light', 'traffic sign',
        'person', 'train', 'truck', 'motorcycle', 'bus', 'rider'
    ]
    ds.classes = {i: name for i, name in enumerate(classes)}
    ds.cls_map = { name: i for i, name in enumerate(classes) if name != '__background__' }
    ds.all_tar_files, ds.all_image_files, ds.all_xml_files, ds.all_db_names = preprocess_udb(udb_path, udb_root)
    return ds


def get_god_dataset(udb_path=None, udb_root=None):
    ds = AttrDict()
    classes = [
        '__background__', 'pedestrian', 'rider', 'car', 'truck', 'box_truck', 'bus'
    ]
    ds.classes = {i: name for i, name in enumerate(classes)}
    ds.cls_map = { name: i for i, name in enumerate(classes) if name != '__background__' }
    ds.all_tar_files, ds.all_image_files, ds.all_xml_files, ds.all_db_names = preprocess_udb(udb_path, udb_root)
    return ds


def get_god_2018_dataset(udb_path=None, udb_root=None):
    ds = AttrDict()
    classes = [
        '__background__', 'pedestrian', 'rider_bicycle', 'rider_bike', 'sedan', 'van',
        'truck', 'box_truck', 'bus', 'sitting_person', 'etc', 'bicycle', 'bike',
        '3-wheels', 'pickup_truck', 'mixer_truck', 'excavator', 'forklift', 'ladder_truck',
        'truck_etc', 'vehicle_etc', 'animal', 'bird'
    ]
    cls_map = { cat_name: cat_name for cat_name in classes if cat_name != '__background__' }
    cls_map['ignored'] = 'etc'
    cls_map['animal_ignored'] = 'etc'
    ds.cls_map_3cls = { 'pedestrian': 'pedestrian', 'sitting_person': 'pedestrian',
        'rider_bicycle': 'rider', 'rider_bike': 'rider', 'bicycle': 'rider', 'bike': 'rider',
        'sedan': 'car', 'van': 'car', 'truck': 'car', 'box_truck': 'car', 'bus': 'car',
        '3-wheels': 'car', 'pickup_truck': 'car', 'mixer_truck': 'car', 'excavator': 'car',
        'forklift': 'car', 'ladder_truck': 'car', 'truck_etc': 'car', 'vehicle_etc': 'car',
    }
    ds.classes = {i: name for i, name in enumerate(classes)}
    ds.cls_map = { key: classes.index(val) for key, val in cls_map.items() }
    ds.all_tar_files, ds.all_image_files, ds.all_xml_files, ds.all_db_names = preprocess_udb(udb_path, udb_root)
    return ds


def get_god_2018_tstl_12cls_dataset(udb_path=None, udb_root=None):
    ds = AttrDict()
    classes = [
        '__background__', 'pedestrian', 'rider', 'vehicle', 'etc', 'two_wheel',
        'ts_circle', 'ts_triangle', 'ts_rectangle', 'ts_etc',
        'tl', 'tl_etc', 'tl_light_only'
    ]
    cls_map = {
        'pedestrian': 'pedestrian',
        'rider_bicycle': 'rider',
        'rider_bike': 'rider',
        'sedan': 'vehicle',
        'van': 'vehicle',
        'truck': 'vehicle',
        'box_truck': 'vehicle',
        'bus': 'vehicle',
        'sitting_person': 'pedestrian',
        'ignored': 'etc',
        'etc': 'etc',
        'bicycle': 'two_wheel',
        'bike': 'two_wheel',
        '3-wheels': 'vehicle',
        'pickup_truck': 'vehicle',
        'mixer_truck': 'vehicle',
        'excavator': 'vehicle',
        'forklift': 'vehicle',
        'ladder_truck': 'vehicle',
        'truck_etc': 'vehicle',
        'vehicle_etc': 'vehicle',
        'ts_circle': 'ts_circle',
        'ts_circle_speed': 'ts_circle',
        'ts_triangle': 'ts_triangle',
        'ts_Inverted_triangle': 'ts_triangle',
        'ts_rectangle': 'ts_rectangle',
        'ts_rectangle_speed': 'ts_rectangle',
        'ts_diamonds': 'ts_rectangle',
        'ts_etc': 'ts_etc',
        'tl_car': 'tl',
        'tl_ped': 'tl',
        'tl_special': 'tl',
        'tl_etc': 'tl_etc',
        'tl_light_only': 'tl_light_only',
    }
    ds.cls_map_3cls = { 'pedestrian': 'pedestrian', 'rider': 'rider', 'vehicle': 'car' }
    ds.classes = {i: name for i, name in enumerate(classes)}
    ds.cls_map = { key: classes.index(val) for key, val in cls_map.items() }
    ds.all_tar_files, ds.all_image_files, ds.all_xml_files, ds.all_db_names = preprocess_udb(udb_path, udb_root)
    return ds


def get_god_2018_6cls_dataset_old(udb_path=None, udb_root=None):
    ds = AttrDict()
    classes = [
        '__background__', 'pedestrian', 'rider', 'car', 'truck', 'bus', 'vehicle'
    ]
    cls_map = {
        'pedestrian': 'pedestrian',
        'rider_bicycle': 'rider',
        'rider_bike': 'rider',
        'sedan': 'car',
        'van': 'car',
        'truck': 'truck',
        'box_truck': 'truck',
        'bus': 'bus',
        'sitting_person': 'pedestrian',
        'etc': '__background__',	
        'bicycle': 'rider',
        'bike': 'rider',
        '3-wheels': 'vehicle',
        'pickup_truck': 'car',
        'mixer_truck': 'truck',
        'excavator': 'vehicle',
        'forklift': 'vehicle',
        'ladder_truck': 'truck',
        'truck_etc': 'truck',
        'vehicle_etc': 'vehicle',
    }
    ds.cls_map_3cls = { 'pedestrian': 'pedestrian', 'rider': 'rider', 'car': 'car', 'truck': 'car', 'bus': 'car', 'vehicle': 'car' }
    ds.classes = {i: name for i, name in enumerate(classes)}
    ds.cls_map = { key: classes.index(val) for key, val in cls_map.items() }
    ds.all_tar_files, ds.all_image_files, ds.all_xml_files, ds.all_db_names = preprocess_udb(udb_path, udb_root)
    return ds


def get_god_2018_11cls_dataset(udb_path=None, udb_root=None):
    ds = AttrDict()
    classes = [
        '__background__', 'pedestrian', 'rider_bicycle', 'rider_bike', 'bicycle', 'bike',
        'car', 'truck', 'bus', 'vehicle', '3-wheels', 'etc'
    ]
    ds.classes_20 = [
        '__background__', 'pedestrian', 'rider_bicycle', 'rider_bike', 'bicycle', 'bike',
        'sedan', 'truck', 'bus', 'excavator', '3-wheels', 'ignored'
    ]
    ds.classes_8 = [
        '__background__', 'pedestrian', 'rider', 'rider', 'rider', 'rider',
        'car', 'truck', 'bus', 'truck', 'rider', 'etc'
    ]
    cls_map = {
        'pedestrian': 'pedestrian',
        'rider_bicycle': 'rider_bicycle',
        'rider_bike': 'rider_bike',
        'sedan': 'car',
        'van': 'car',
        'truck': 'truck',
        'box_truck': 'truck',
        'bus': 'bus',
        'sitting_person': 'pedestrian',
        'ignored': 'etc',
        'etc': 'etc',
        'bicycle': 'bicycle',
        'bike': 'bike',
        '3-wheels': '3-wheels',
        'pickup_truck': 'car',
        'mixer_truck': 'truck',
        'excavator': 'vehicle',
        'forklift': 'vehicle',
        'ladder_truck': 'truck',
        'truck_etc': 'truck',
        'vehicle_etc': 'truck',
    }
    ds.cls_map_3cls = { 'pedestrian': 'pedestrian',
        'rider_bicycle': 'rider', 'rider_bike': 'rider', 'bicycle': 'rider', 'bike': 'rider',
        'car': 'car', 'truck': 'car', 'bus': 'car', 'vehicle': 'car', '3-wheels': 'car',
    }
    ds.classes = {i: name for i, name in enumerate(classes)}
    ds.cls_map = { key: classes.index(val) for key, val in cls_map.items() }
    ds.all_tar_files, ds.all_image_files, ds.all_xml_files, ds.all_db_names = preprocess_udb(udb_path, udb_root)
    return ds


def get_god_2018_5cls_dataset(udb_path=None, udb_root=None):
    ds = AttrDict()
    classes = [
        '__background__', 'pedestrian', 'rider', '2-wheels', 'vehicle', 'etc'
    ]
    cls_map = {
        'pedestrian': 'pedestrian',
        'rider_bicycle': 'rider',
        'rider_bike': 'rider',
        'sedan': 'vehicle',
        'van': 'vehicle',
        'truck': 'vehicle',
        'box_truck': 'vehicle',
        'bus': 'vehicle',
        'sitting_person': 'pedestrian',
        'ignored': 'etc',	
        'etc': 'etc',	
        'bicycle': '2-wheels',
        'bike': '2-wheels',
        '3-wheels': 'vehicle',
        'pickup_truck': 'vehicle',
        'mixer_truck': 'vehicle',
        'excavator': 'vehicle',
        'forklift': 'vehicle',
        'ladder_truck': 'vehicle',
        'truck_etc': 'vehicle',
        'vehicle_etc': 'vehicle',
    }
    ds.cls_map_3cls = { 'pedestrian': 'pedestrian', 'rider': 'rider', '2-wheels': 'rider', 'vehicle': 'car' }
    ds.classes = {i: name for i, name in enumerate(classes)}
    ds.cls_map = { key: classes.index(val) for key, val in cls_map.items() }
    ds.all_tar_files, ds.all_image_files, ds.all_xml_files, ds.all_db_names = preprocess_udb(udb_path, udb_root)
    return ds


def get_god_2018_4cls_dataset_old(udb_path=None, udb_root=None):
    ds = AttrDict()
    classes = [
        '__background__', 'pedestrian', 'rider', 'vehicle', 'etc'
    ]
    cls_map = {
        'pedestrian': 'pedestrian',
        'rider_bicycle': 'rider',
        'rider_bike': 'rider',
        'sedan': 'vehicle',
        'van': 'vehicle',
        'truck': 'vehicle',
        'box_truck': 'vehicle',
        'bus': 'vehicle',
        'sitting_person': 'pedestrian',
        'etc': 'etc',	
        'bicycle': 'rider',
        'bike': 'rider',
        '3-wheels': 'vehicle',
        'pickup_truck': 'vehicle',
        'mixer_truck': 'vehicle',
        'excavator': 'vehicle',
        'forklift': 'vehicle',
        'ladder_truck': 'vehicle',
        'truck_etc': 'vehicle',
        'vehicle_etc': 'vehicle',
    }
    ds.cls_map_3cls = { 'pedestrian': 'pedestrian', 'rider': 'rider', 'vehicle': 'car' }
    ds.classes = {i: name for i, name in enumerate(classes)}
    ds.cls_map = { key: classes.index(val) for key, val in cls_map.items() }
    ds.all_tar_files, ds.all_image_files, ds.all_xml_files, ds.all_db_names = preprocess_udb(udb_path, udb_root)
    return ds


def get_god_2018_4cls_dataset(udb_path=None, udb_root=None):
    ds = AttrDict()
    classes = [
        '__background__', 'pedestrian', 'rider', 'vehicle', 'etc'
    ]
    cls_map = {
        'pedestrian': 'pedestrian',
        'rider_bicycle': 'rider',
        'rider_bike': 'rider',
        'sedan': 'vehicle',
        'van': 'vehicle',
        'truck': 'vehicle',
        'box_truck': 'vehicle',
        'bus': 'vehicle',
        'sitting_person': 'pedestrian',
        'ignored': 'etc',
        'etc': 'etc',
        'bicycle': 'etc',
        'bike': 'etc',
        '3-wheels': 'rider',
        'pickup_truck': 'vehicle',
        'mixer_truck': 'vehicle',
        'excavator': 'vehicle',
        'forklift': 'vehicle',
        'ladder_truck': 'vehicle',
        'truck_etc': 'vehicle',
        'vehicle_etc': 'vehicle',
        'animal': 'etc',
        'bird': 'etc',
        'animal_ignored': 'etc',
    }
    ds.cls_map_3cls = { 'pedestrian': 'pedestrian', 'rider': 'rider', 'vehicle': 'car' }
    ds.classes = {i: name for i, name in enumerate(classes)}
    ds.cls_map = { key: classes.index(val) for key, val in cls_map.items() }
    ds.all_tar_files, ds.all_image_files, ds.all_xml_files, ds.all_db_names = preprocess_udb(udb_path, udb_root)
    return ds


def get_god_2018_3cls_dataset(udb_path=None, udb_root=None):
    ds = AttrDict()
    classes = [
        '__background__', 'ped_rider', 'car', 'etc'
    ]
    cls_map = {
        'pedestrian': 'ped_rider',
        'rider_bicycle': 'ped_rider',
        'rider_bike': 'ped_rider',
        'sedan': 'car',
        'van': 'car',
        'truck': 'car',
        'box_truck': 'car',
        'bus': 'car',
        'sitting_person': 'ped_rider',
        'ignored': 'etc',
        'etc': 'etc',
        'bicycle': 'etc',
        'bike': 'etc',
        '3-wheels': 'ped_rider',
        'pickup_truck': 'car',
        'mixer_truck': 'car',
        'excavator': 'car',
        'forklift': 'car',
        'ladder_truck': 'car',
        'truck_etc': 'car',
        'vehicle_etc': 'car',
        'animal': 'etc',
        'bird': 'etc',
        'animal_ignored': 'etc',
    }
    ds.cls_map_3cls = { 'ped_rider': 'pedestrian', 'car': 'car' }
    ds.classes = {i: name for i, name in enumerate(classes)}
    ds.cls_map = { key: classes.index(val) for key, val in cls_map.items() }
    ds.all_tar_files, ds.all_image_files, ds.all_xml_files, ds.all_db_names = preprocess_udb(udb_path, udb_root)
    return ds


def get_god_2018_6cls_dataset(udb_path=None, udb_root=None):
    ds = AttrDict()
    classes = [
        '__background__', 'pedestrian', 'rider', 'car', 'truck', 'bus', 'etc'
    ]
    cls_map = {
        'pedestrian': 'pedestrian',
        'rider_bicycle': 'rider',
        'rider_bike': 'rider',
        'sedan': 'car',
        'van': 'car',
        'truck': 'truck',
        'box_truck': 'truck',
        'bus': 'bus',
        'sitting_person': 'pedestrian',
        'ignored': 'etc',
        'etc': 'etc',
        'bicycle': 'etc',
        'bike': 'etc',
        '3-wheels': 'rider',
        'pickup_truck': 'car',
        'mixer_truck': 'truck',
        'excavator': 'truck',
        'forklift': 'truck',
        'ladder_truck': 'truck',
        'truck_etc': 'truck',
        'vehicle_etc': 'truck',
        'animal': 'etc',
        'bird': 'etc',
        'animal_ignored': 'etc',
    }
    ds.cls_map_3cls = { 'pedestrian': 'pedestrian', 'rider': 'rider', 'car': 'car', 'truck': 'car', 'bus': 'car' }
    ds.classes = {i: name for i, name in enumerate(classes)}
    ds.cls_map = { key: classes.index(val) for key, val in cls_map.items() }
    ds.all_tar_files, ds.all_image_files, ds.all_xml_files, ds.all_db_names = preprocess_udb(udb_path, udb_root)
    return ds


def get_god_2018_8cls_dataset(udb_path=None, udb_root=None):
    ds = AttrDict()
    classes = [
        '__background__', 'pedestrian', 'rider', 'car', 'van', 'truck', 'box_truck', 'bus', 'etc'
    ]
    cls_map = {
        'pedestrian': 'pedestrian',
        'rider_bicycle': 'rider',
        'rider_bike': 'rider',
        'sedan': 'car',
        'van': 'van',
        'truck': 'truck',
        'box_truck': 'box_truck',
        'bus': 'bus',
        'sitting_person': 'pedestrian',
        'ignored': 'etc',
        'etc': 'etc',
        'bicycle': 'etc',
        'bike': 'etc',
        '3-wheels': 'rider',
        'pickup_truck': 'car',
        'mixer_truck': 'truck',
        'excavator': 'truck',
        'forklift': 'truck',
        'ladder_truck': 'truck',
        'truck_etc': 'truck',
        'vehicle_etc': 'truck',
    }
    ds.cls_map_3cls = { 'pedestrian': 'pedestrian', 'rider': 'rider', 'car': 'car', 'van': 'car', 'truck': 'car', 'box_truck': 'car', 'bus': 'car' }
    ds.classes = {i: name for i, name in enumerate(classes)}
    ds.cls_map = { key: classes.index(val) for key, val in cls_map.items() }
    ds.all_tar_files, ds.all_image_files, ds.all_xml_files, ds.all_db_names = preprocess_udb(udb_path, udb_root)
    return ds


def get_god_2018_9cls_dataset(udb_path=None, udb_root=None):
    ds = AttrDict()
    classes = [
        '__background__', 'pedestrian', 'rider', 'car', 'van', 'truck', 'box_truck', 'bus', 'animal', 'etc'
    ]
    cls_map = {
        'pedestrian': 'pedestrian',
        'rider_bicycle': 'rider',
        'rider_bike': 'rider',
        'sedan': 'car',
        'van': 'van',
        'truck': 'truck',
        'box_truck': 'box_truck',
        'bus': 'bus',
        'sitting_person': 'pedestrian',
        'ignored': 'etc',
        'etc': 'etc',
        'bicycle': 'etc',
        'bike': 'etc',
        '3-wheels': 'rider',
        'pickup_truck': 'car',
        'mixer_truck': 'truck',
        'excavator': 'truck',
        'forklift': 'truck',
        'ladder_truck': 'truck',
        'truck_etc': 'truck',
        'vehicle_etc': 'truck',
        'animal': 'animal',
        'bird': 'etc',
        'animal_ignored': 'etc',
    }
    ds.cls_map_3cls = { 'pedestrian': 'pedestrian', 'rider': 'rider', 'car': 'car', 'van': 'car', 'truck': 'car', 'box_truck': 'car', 'bus': 'car' }
    ds.classes = {i: name for i, name in enumerate(classes)}
    ds.cls_map = { key: classes.index(val) for key, val in cls_map.items() }
    ds.all_tar_files, ds.all_image_files, ds.all_xml_files, ds.all_db_names = preprocess_udb(udb_path, udb_root)
    return ds


def get_god_2018_coco_dataset(udb_path=None, udb_root=None):
    ds = AttrDict()
    classes = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
        'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush',
        'rider_bicycle', 'rider_bike', 'rider', '3-wheels', 'vehicle',
    ]
    cls_map = {
        'pedestrian': 'person',
        'rider_bicycle': 'rider_bicycle',
        'rider_bike': 'rider_bike',
        'sedan': 'car',
        'van': 'bus',
        'truck': 'truck',
        'box_truck': 'truck',
        'bus': 'bus',
        'sitting_person': 'person',
        'etc': '__background__',
        'bicycle': 'bicycle',
        'bike': 'motorcycle',
        '3-wheels': '3-wheels',
        'pickup_truck': 'truck',
        'mixer_truck': 'truck',
        'excavator': 'vehicle',
        'forklift': 'vehicle',
        'ladder_truck': 'truck',
        'truck_etc': 'truck',
        'vehicle_etc': 'truck',
        'rider': 'rider',
    }
    ds.classes = {i: name for i, name in enumerate(classes)}
    ds.cls_map = { key: classes.index(val) for key, val in cls_map.items() }
    ds.all_tar_files, ds.all_image_files, ds.all_xml_files, ds.all_db_names = preprocess_udb(udb_path, udb_root)

    ds.cls_map_4cls = [0, ds.cls_map['pedestrian'], ds.cls_map['rider'], ds.cls_map['sedan'], ds.cls_map['etc']]
    ds.cls_map_4cls_inv = { val: idx for idx, val in enumerate(ds.cls_map_4cls) if idx > 0 }
    ds.cls_map_4cls_inv[0] = 0

    ds.cls_map_5cls = [0, ds.cls_map['pedestrian'], ds.cls_map['rider'], ds.cls_map['bicycle'], ds.cls_map['sedan'], ds.cls_map['etc']]
    ds.cls_map_5cls_inv = { val: idx for idx, val in enumerate(ds.cls_map_5cls) if idx > 0 }
    ds.cls_map_5cls_inv[0] = 0

    ds.cls_map_8cls = [0, ds.cls_map['pedestrian'], ds.cls_map['rider'], ds.cls_map['sedan'], ds.cls_map['van'], ds.cls_map['truck'],
        ds.cls_map['box_truck'], ds.cls_map['bus'], ds.cls_map['etc']]
    ds.cls_map_8cls_inv = { val: idx for idx, val in enumerate(ds.cls_map_8cls) if idx > 0 }
    ds.cls_map_8cls_inv[0] = 0

    ds.cls_map_11cls = [0, ds.cls_map['pedestrian'], ds.cls_map['rider_bicycle'], ds.cls_map['rider_bike'], ds.cls_map['bicycle'], ds.cls_map['bike'],
        ds.cls_map['sedan'], ds.cls_map['truck'], ds.cls_map['bus'], ds.cls_map['excavator'], ds.cls_map['3-wheels'], ds.cls_map['etc']]
    ds.cls_map_11cls_inv = { val: idx for idx, val in enumerate(ds.cls_map_11cls) if idx > 0 }
    ds.cls_map_11cls_inv[0] = 0

    return ds


def get_god_2020_10cls_dataset(udb_path=None, udb_root=None):
    ds = AttrDict()
    classes = [
        '__background__', 'pedestrian', 'rider', 'car', 'truck', 'bus', 'ts_circle', 'ts_triangle', 'ts_rectangle', 'tl', 'etc'
    ]
    cls_map = {
        '3-wheels': 'rider',
        'animal': 'etc',
        'animal_ignored': 'etc',
        'bicycle': 'etc',
        'bike': 'etc',
        'bird': 'etc',
        'box_truck': 'truck',
        'bus': 'bus',
        'car': 'car',
        'excavator': 'truck',
        'forklift': 'truck',
        'ignored': 'etc',
        'Ignored': 'etc',   ###
        'ladder_truck': 'truck',
        'mixer_truck': 'truck',
        'pedestrian': 'pedestrian',
        'pickup_truck': 'truck',
        'rider': 'rider',
        'rider_bicycle': 'rider',
        'rider_bike': 'rider',
        'sedan': 'car',
        'sitting_person': 'pedestrian',
        'special_vehicle': 'truck',
        'tl_car': 'tl',
        'tl_vehicle': 'tl',
        'TL_vehicle': 'tl',     ###
        'tl_ignored': 'etc',
        'TL_ignored': 'etc',    ###
        'tl_light_only': 'etc',
        'tl_ped': 'tl',
        'tl_special': 'etc',
        'truck': 'truck',
        'truck_etc': 'truck',
        'ts_circle': 'ts_circle',
        'TS_circle': 'ts_circle',   ###
        'ts_circle_speed': 'ts_circle',
        'ts_diamonds': 'ts_rectangle',
        'ts_ignored': 'etc',
        'TS_ignored': 'etc',    ###
        'ts_Inverted_triangle': 'ts_triangle',
        'ts_rectangle': 'ts_rectangle',
        'ts_square': 'ts_rectangle',
        'TS_square': 'ts_rectangle',    ###
        'ts_rectangle_speed': 'ts_rectangle',
        'ts_supplementary': 'etc',
        'ts_triangle': 'ts_triangle',
        'TS_triangle': 'ts_triangle',   ###
        'van': 'car',
        'vehicle_etc': 'truck',
        'false_positive': '__background__',
    }

    ds.cls_map_5cls = { 'pedestrian': 'pedestrian', 'rider': 'rider', 'car': 'car', 'truck': 'car', 'bus': 'car', 'ts_circle': 'ts', 'ts_triangle': 'ts', 'ts_rectangle': 'ts', 'tl': 'tl'}
    ds.classes = {i: name for i, name in enumerate(classes)}
    ds.cls_map = { key: classes.index(val) for key, val in cls_map.items() }
    ds.all_tar_files, ds.all_image_files, ds.all_xml_files, ds.all_db_names = preprocess_udb(udb_path, udb_root)
    return ds


def get_god_2020_20cls_dataset(udb_path=None, udb_root=None):
    ds = AttrDict()
    classes = [
        '__background__',
        'pedestrian',
        'rider_bicycle',
        'rider_bike',        
        '3-wheels',
        'sedan',
        'van',
        'truck',
        'bus',
        'excavator',
        'forklift',
        'ts_circle',
        'ts_triangle',
        'ts_rectangle',
        'ts_sup_arrow',
        'ts_sup_drawing',
        'ts_sup_letter',
        'ts_sup_zone',
        'tl_ped',
        'tl_car',
        'etc'
    ]
    cls_map = {
        '3-wheels': '3-wheels',
        '3-wheels_rider': '3-wheels',
        'animal': 'etc',
        'animal_ignored': 'etc',
        'bicycle': 'etc',
        'bike': 'etc',
        'bird': 'etc',
        'box_truck': 'truck',
        'bus': 'bus',
        'car': 'etc',       # ?
        'excavator': 'excavator',
        'forklift': 'forklift',
        'ignored': 'etc',
        'Ignored': 'etc',   ###
        'ladder_truck': 'truck',
        'mixer_truck': 'truck',
        'pedestrian': 'pedestrian',
        'pickup_truck': 'truck',
        'rider': 'etc',
        'rider_bicycle': 'rider_bicycle',
        'rider_bike': 'rider_bike',
        'sedan': 'sedan',
        'sitting_person': 'pedestrian',
        'special_vehicle': 'truck',
        'trailer': 'etc',
        'tl_car': 'tl_car',
        'tl_vehicle': 'tl_car',
        'TL_vehicle': 'tl_car',     ###
        'tl_ignored': 'etc',
        'TL_ignored': 'etc',    ###
        'tl_light_only': 'etc',
        'tl_ped': 'tl_ped',
        'tl_special': 'etc',
        'truck': 'truck',
        'truck_etc': 'truck',
        'ts_circle': 'ts_circle',
        'TS_circle': 'ts_circle',   ###
        'ts_circle_speed': 'ts_circle',
        'ts_diamonds': 'ts_rectangle',
        'ts_ignored': 'etc',
        'TS_ignored': 'etc',    ###
        'ts_Inverted_triangle': 'ts_triangle',
        'ts_rectangle': 'ts_rectangle',
        'ts_square': 'ts_rectangle',
        'TS_square': 'ts_rectangle',    ###
        'ts_rectangle_speed': 'ts_rectangle',
        'ts_rectangle_arrow': 'ts_rectangle',
        'ts_supplementary': 'etc',
        'ts_sup_arrow': 'ts_sup_arrow',
        'ts_sup_drawing': 'ts_sup_drawing',
        'ts_sup_letter': 'ts_sup_letter',
        'ts_sup_zone': 'ts_sup_zone',
        'ts_sup_ignored': 'etc',
        'ts_triangle': 'ts_triangle',
        'TS_triangle': 'ts_triangle',   ###
        'ts_rear': 'etc',
        'tl_rear': 'etc',
        'ts_main_zone': 'etc',        
        'van': 'van',
        'vehicle_etc': 'truck',
        'ehicle_special': 'etc',
        'false_positive': '__background__',
    }

    ds.cls_map_5cls = { 'pedestrian': 'pedestrian', 'rider': 'rider', 'car': 'car', 'truck': 'car', 'bus': 'car', 'ts_circle': 'ts', 'ts_triangle': 'ts', 'ts_rectangle': 'ts', 'tl': 'tl'}
    ds.classes = {i: name for i, name in enumerate(classes)}
    ds.cls_map = { key: classes.index(val) for key, val in cls_map.items() }
    ds.all_tar_files, ds.all_image_files, ds.all_xml_files, ds.all_db_names = preprocess_udb(udb_path, udb_root)
    return ds


def get_god_sod_2021_52cls_dataset(udb_path=None, udb_root=None):
    ds = AttrDict()
    classes = [
        '__background__', 
        'pedestrian',
		'sitting_person', 		
		'rider_bike', 
        'rider_bicycle', 
        'bicycle', 
		'bike',		
		'3-wheels',
        '3-wheels_rider',		
        'sedan', 
        'van',         
        'truck',        
		'box_truck', 
		'pickup_truck',		
		'mixer_truck',
		'ladder_truck', 
        'bus',		
		'trailer',                 
		'excavator',
		'forklift',		
		'ts_triangle',
		'ts_Inverted_triangle', 		
		'ts_circle',
		'ts_circle_speed', 
		'ts_rectangle', 
        'ts_rectangle_speed', 
		'ts_rectangle_arrow', 
        'ts_diamonds',         		
        'ts_rear',
		'ts_main_zone', 
		'ts_sup_letter',
        'ts_sup_arrow',
		'ts_sup_zone',
		'ts_sup_drawing', 		        
        'tl_car',
		'tl_ped',
        'tl_rear',
		'tl_light_only', 	
        'blocking_bar', 		
        'obstacle_drum', 
        'obstacle_cone', 
        'obstacle_bollard_cylinder', 
		'obstacle_bollard_marker',
		'obstacle_bollard_stone',
		'obstacle_bollard_U_shaped',
		'obstacle_bollard_barricade', 
        'obstacle_cylinder',
        'parking_sign',   
		'parking_lock', 		
        'parking_cylinder',
        'parking_stopper_marble', 
        'parking_stopper_bar',
        'parking_stopper_separated',
        'etc'
    ]
    cls_map = {
        'sedan': 'sedan', 
        'van': 'van', 
        'pedestrian': 'pedestrian', 
        'box_truck': 'box_truck', 
        'ignored': 'etc', 
        'ts_circle': 'ts_circle', 
        'tl_car': 'tl_car',
        'tl_ignored': 'etc', 
        'ts_ignored': 'etc', 
        'ts_rear': 'ts_rear', 
        'ts_rectangle': 'ts_rectangle', 
        'rider_bike': 'rider_bike', 
        'truck': 'truck', 
        'pickup_truck': 'pickup_truck', 
        'ts_rectangle_speed': 'ts_rectangle_speed', 
        'bus': 'bus', 
        'ts_diamonds': 'ts_diamonds', 
        'ts_sup_ignored': 'etc', 
        'obstacle_ignored': 'etc', 
        'tl_rear': 'tl_rear', 
        'vehicle_etc': 'etc', 
        'rider_bicycle': 'rider_bicycle', 
        'bicycle': 'bicycle', 
        'tl_ped': 'tl_ped', 
        'obstacle_cone': 'obstacle_cone', 
        'ehicle_special': 'etc', 
        'ts_triangle': 'ts_triangle', 
        'ts_sup_letter': 'ts_sup_letter', 
        'truck_etc': 'etc', 
        'sitting_person': 'sitting_person',
        'ts_sup_arrow': 'ts_sup_arrow', 
        'bird': '__background__', 
        'animal_ignored': '__background__', 
        'obstacle_bollard_special': 'etc', 
        'obstacle_cylinder': 'obstacle_cylinder', 
        'bike': 'bike', 
        'animal': '__background__', 
        'trailer': 'trailer', 
        'ts_Inverted_triangle': 'ts_Inverted_triangle', 
        'tl_light_only': 'tl_light_only', 
        'obstacle_bollard_cylinder': 'obstacle_bollard_cylinder', 
        'ts_rectangle_arrow': 'ts_rectangle_arrow', 
        'parking_ignored': 'etc', 
        'parking_stopper_marble': 'parking_stopper_marble', 
        'blocking_ignored': 'etc', 
        'obstacle_bollard_marker': 'obstacle_bollard_marker', 
        'tl_special': 'etc', 
        'excavator': 'excavator', 
        'obstacle_bollard_stone': 'obstacle_bollard_stone', 
        'forklift': 'forklift', 
        'parking_sign': 'parking_sign', 
        'mixer_truck': 'mixer_truck', 
        'parking_cylinder': 'parking_cylinder', 
        '3-wheels': '3-wheels', 
        'obstacle_bollard_U_shaped': 'obstacle_bollard_U_shaped', 
        '3-wheels_rider': '3-wheels_rider', 
        'parking_stopper_bar': 'parking_stopper_bar', 
        'ladder_truck': 'ladder_truck', 
        'parking_stopper_separated': 'parking_stopper_separated', 
        'obstacle_drum': 'obstacle_drum', 
        'obstacle_bollard_barricade': 'obstacle_bollard_barricade', 
        'ts_sup_drawing': 'ts_sup_drawing', 
        'blocking_bar': 'blocking_bar', 
        'parking_special': 'etc', 
        'ts_main_zone': 'ts_main_zone', 
        'parking_lock': 'parking_lock', 
        'ts_circle_speed': 'ts_circle_speed', 
        'blocking_special': 'etc', 
        'false_positive': '__background__', 
        'ts_sup_zone': 'ts_sup_zone', 
        'vehicle_special': 'etc',
        ##
        'ts_supplementary': 'etc',
        'rider': 'etc',
        'car': 'etc',
        'TS_ignored': 'etc',
        'TL_vehicle': 'etc',
        'TL_ignored': 'etc',
        'TS_triangle': 'etc',
        'TS_triangle': 'etc',
    }

    ds.classes = {i: name for i, name in enumerate(classes)}
    ds.cls_map = { key: classes.index(val) for key, val in cls_map.items() }
    ds.all_tar_files, ds.all_image_files, ds.all_xml_files, ds.all_db_names = preprocess_udb(udb_path, udb_root)
    return ds


get_dataset_fn_dict = {
    'coco': get_coco_dataset,
    'cityscapes': get_cityscapes_dataset,
    'god': get_god_dataset,
    'god_2018': get_god_2018_dataset,
    'god_2018_tstl_12cls': get_god_2018_tstl_12cls_dataset,
    'god_2018_11cls': get_god_2018_11cls_dataset,
    'god_2018_9cls': get_god_2018_9cls_dataset,
    'god_2018_8cls': get_god_2018_8cls_dataset,
    'god_2018_6cls': get_god_2018_6cls_dataset,
    'god_2018_5cls': get_god_2018_5cls_dataset,
    'god_2018_4cls': get_god_2018_4cls_dataset,
    'god_2018_3cls': get_god_2018_3cls_dataset,
    'god_2018_coco': get_god_2018_coco_dataset,
    'god_2020_10cls': get_god_2020_10cls_dataset,
    'god_2020_20cls': get_god_2020_20cls_dataset,
    'god_sod_2021_52cls': get_god_sod_2021_52cls_dataset,
}


def get_dataset(dataset, *args, **kwargs):
    return get_dataset_fn_dict[dataset](*args, **kwargs)


def preprocess_udb(udb_path, udb_root=None):
    def get_filename_wo_ext(path):
        name_wo_path = path.strip().split('/')[-1]
        name_wo_ext = '.'.join(name_wo_path.split('.')[:-1])
        return name_wo_ext

    all_image_files = []
    all_xml_files = []
    all_tar_files = []
    all_db_names = []

    if udb_path is None:
        return all_tar_files, all_image_files, all_xml_files, all_db_names

    if udb_root is None:
        udb_root = '/'.join(udb_path.strip().split('/')[:-1]) + '/../'

    for line in open(udb_path, 'r'):
        udb_name_wo_ext, udb_image_set = line.strip().split(':')
        tar_path = os.path.join(udb_root, udb_name_wo_ext + '.tar')
        tar_db = tarfile.open(tar_path, 'r')
        all_files = tar_db.getmembers()
        tar_db.close()

        tar_db = tarfile.open(tar_path, 'r')
        image_set_files = [f for f in all_files if f.name.startswith('ImageSets') and f.name.strip().split('/')[-1] == udb_image_set + '.txt']
        image_files_dict = {}
        for image_set_file in image_set_files:
            image_files_dict.update({
                image_file_wo_ext.strip(): None
                for image_file_wo_ext in tar_db.extractfile(image_set_file)
            })
        tar_db.close()

        image_files = [f for f in all_files if f.name.startswith('JPEGImages') and image_files_dict.has_key(get_filename_wo_ext(f.name))]
        for f_idx, f in enumerate(image_files):
            image_files_dict[get_filename_wo_ext(f.name)] = f_idx
        for key, val in image_files_dict.items():
            if val is None:
                print('[ERROR] File {} not found in {}'.format(key, tar_path))

        xml_files = [None] * len(image_files)
        for f in all_files:
            if f.name.startswith('Annotations') and image_files_dict.has_key(get_filename_wo_ext(f.name)):
                f_idx = image_files_dict[get_filename_wo_ext(f.name)]
                xml_files[f_idx] = f
        for item in xml_files:
            if item is None:
                print('[ERROR] File {} not found in {}'.format(item, tar_path))

        all_tar_files.append(tar_path)
        all_image_files.append(image_files)
        all_xml_files.append(xml_files)
        all_db_names.append(udb_name_wo_ext)

    return all_tar_files, all_image_files, all_xml_files, all_db_names


def load_udb_images_from_dataset(ds, start_idx=0, load_img=True):
    return load_udb_images(ds.all_tar_files, ds.all_image_files, ds.all_xml_files, ds.all_db_names, start_idx, load_img)


def load_udb_images(all_tar_files, all_image_files, all_xml_files, all_db_names, start_idx, load_img):
    def make_ann(node):
        if node.text is not None:
            val = node.text.strip()
            if len(val) > 0:
                return val

        subtree = {}
        for child in node:
            val = subtree.get(child.tag)
            if val is None:
                subtree[child.tag] = make_ann(child)
            elif val.__class__ is list:
                subtree[child.tag].extend(make_ann(child))
            else:
                raise Exception('Wrong annotation format: {}'.format(xml_file.name))
        if node.tag in ['object', 'part', 'size']:
            return [subtree]
        return subtree

    i = 0
    for tar_path, image_files, xml_files, db_name in zip(all_tar_files, all_image_files, all_xml_files, all_db_names):
        tar_db = tarfile.open(tar_path, 'r')
        for image_file, xml_file in zip(image_files, xml_files):
            img_path = image_file.name

            if i < start_idx:
                img = None
                ann = None
                xml_path = None
                width = 0
                height = 0

            else:
                if load_img:
                    img_bin = tar_db.extractfile(image_file).read()
                    img = cv2.imdecode(np.fromstring(img_bin, dtype=np.uint8), cv2.IMREAD_COLOR)
                    img_path = image_file.name
                else:
                    img = None

                if xml_file is not None:
                    ann_bin = tar_db.extractfile(xml_file).read()
                    ann = make_ann(ET.fromstring(ann_bin))
                    xml_path = xml_file.name
                    if not ann.has_key('size') or len(ann['size']) != 1:
                        raise Exception('Wrong annotation format: {}'.format(xml_file.name))
                    width = int(ann['size'][0]['width'])
                    height = int(ann['size'][0]['height'])
                    if load_img and (img.shape[1] != width or img.shape[0] != height):
                        raise Exception('Size mismatch: {}x{} (image) != {}x{} (xml)'.format(img.shape[1], img.shape[0], width, height))
                else:
                    ann = None
                    xml_path = None
                    width = 0
                    height = 0

            i += 1
            yield img, width, height, ann, img_path, xml_path, tar_path, db_name
        tar_db.close()
