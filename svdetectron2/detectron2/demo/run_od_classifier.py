# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer

from evaluations.detectionbox import DetectionBox
# import pickle
import _pickle
import numpy as np

import sys

from train_net import Trainer
#from tools.train_net import Trainer
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import GeneralizedRCNNWithTTA
from collections import OrderedDict
import detectron2.data.transforms as T
from detectron2.data import MetadataCatalog
import torch

from detectron2.utils.inference_util import write_xml, write_xml2

import xml.etree.ElementTree as ET
from detectron2.data.datasets import udb_datasets 
from detectron2.structures import Boxes, Instances

import math


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--input-xml",
        nargs="+",
        help="A list of space separated input xmls; "
        "or a single glob pattern such as 'directory/*.xml'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--model_type", help="ffc or svm"
    )
    
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument("--proposal-iou", action="store_true", help="extract proposal iou distribution.")
    # parser.add_argument("--save-outimg", action="store_true", help="save output image.")
    parser.add_argument("--save-outtxt", action="store_true", help="save output textfile.")
    parser.add_argument(
        "--save-outimg",
        type=int,
        default=0,
        help="saving detection output image",
    )
    
    return parser


def load_udb_annotation(xml_file):
    cls_dict = udb_datasets.get_god_sod_2021_52cls_dataset()

    tree = ET.parse(xml_file)

    # r = {
    #         "file_name": jpeg_file,
    #         "image_id": fileid,
    #         "height": int(tree.findall("./size/height")[0].text),
    #         "width": int(tree.findall("./size/width")[0].text),
    # }
    r = {}
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
        # bbox[0] -= 1.0
        # bbox[1] -= 1.0
        instances.append(
            {"category_id": cls_dict.cls_map[cls] - 1, "gt_bbox": bbox}
        )
    # r["annotations"] = instances

    return instances



def get_max_iou(pred_boxes, gt_box):
    """
    calculate the iou multiple pred_boxes and 1 gt_box (the same one)
    pred_boxes: multiple predict  boxes coordinate
    gt_box: ground truth bounding  box coordinate
    return: the max overlaps about pred_boxes and gt_box
    """
    # 1. calculate the inters coordinate
    if pred_boxes.shape[0] > 0:
        ixmin = np.maximum(pred_boxes[:, 0], gt_box[0])
        ixmax = np.minimum(pred_boxes[:, 2], gt_box[2])
        iymin = np.maximum(pred_boxes[:, 1], gt_box[1])
        iymax = np.minimum(pred_boxes[:, 3], gt_box[3])

        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)

    # 2.calculate the area of inters
        inters = iw * ih

    # 3.calculate the area of union
        uni = ((pred_boxes[:, 2] - pred_boxes[:, 0] + 1.) * (pred_boxes[:, 3] - pred_boxes[:, 1] + 1.) +
               (gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.) -
               inters)

    # 4.calculate the overlaps and find the max overlap ,the max overlaps index for pred_box
        iou = inters / uni
        iou_max = np.max(iou)
        nmax = np.argmax(iou)
        return iou, iou_max, nmax


def get_color(_cls):
    color = (0, 0, 255)   # bg
    # if _cls == 0:                   # pedestrian
    #     color = (75, 25, 230)
    # elif _cls == 1:                 # sitting_person
    #     color = (60, 245, 210)
    # elif _cls == 2:                 # rider_bike
    #     color = (25, 255, 255)
    # elif _cls == 3:                 # rider_bicycle
    #     color = (75, 180, 60)
    # elif _cls == 4:                 # bicycle
    #     color = (128, 128, 0)
    # elif _cls == 5:                 # bike
    #     color = (255, 190, 230)
    # elif _cls == 6:                 # 3-wheels
    #     color = (40, 110, 170)
    # elif _cls == 7:                 # 3-wheels_rider
    #     color = (, 128, 255)
    # elif _cls == 8:                 # sedan
    #     color = (52, 226, 13)
    # elif _cls == 9:                 # van
    #     color = (52, 226, 13)
    # elif _cls == 10:                # truck
    #     color = (52, 226, 13)
    # elif _cls == 11:                # box_truck
    #     color = (79, 233, 252)
    # elif _cls == 12:                # pickup_truck
    #     color = (41, 41, 239)
    # elif _cls == 13:                # mixer_truck
    #     color = (17, 125, 193)
    # elif _cls == 14:                # ladder_truck
    #     color = (207, 159, 114)
    # elif _cls == 15:                # bus
    #     color = (79, 233, 252)
    # elif _cls == 16:                # trailer
    #     color = (255, 153, 51)
    # elif _cls == 17:                # excavator
    #     color = (0, 128, 255)
    # elif _cls == 18:                # forklift
    #     color = (52, 226, 13)
    # elif _cls == 19:                # ts_triangle
    #     color = (52, 226, 13)
    # elif _cls == 20:                # ts_Inverted_triangle
    #     color = (52, 226, 13)
    # elif _cls == 21:                # ts_circle
    #     color = (79, 233, 252)
    # elif _cls == 22:                # ts_circle_speed
    #     color = (41, 41, 239)
    # elif _cls == 23:                # ts_rectangle
    #     color = (17, 125, 193)
    # elif _cls == 24:                # ts_rectangle_speed
    #     color = (207, 159, 114)
    # elif _cls == 25:                # ts_rectangle_arrow
    #     color = (79, 233, 252)
    # elif _cls == 26:                # ts_diamonds
    #     color = (255, 153, 51)
    # elif _cls == 27:                # ts_rear
    #     color = (0, 128, 255)
    # elif _cls == 28:                # ts_main_zone
    #     color = (52, 226, 13)
    # elif _cls == 29:                # ts_sup_letter
    #     color = (52, 226, 13)
    # elif _cls == 30:                # ts_sup_arrow
    #     color = (52, 226, 13)
    # elif _cls == 31:                # ts_sup_zone
    #     color = (79, 233, 252)
    # elif _cls == 32:                # ts_sup_drawing
    #     color = (41, 41, 239)
    # elif _cls == 33:                # tl_car
    #     color = (17, 125, 193)
    # elif _cls == 34:                # tl_ped
    #     color = (207, 159, 114)
    # elif _cls == 35:                # tl_rear
    #     color = (79, 233, 252)
    # elif _cls == 36:                # tl_light_only
    #     color = (255, 153, 51)
    # elif _cls == 37:                # blocking_bar
    #     color = (0, 128, 255)
    # elif _cls == 38:                # obstacle_drum
    #     color = (52, 226, 13)
    # elif _cls == 39:                # obstacle_cone
    #     color = (52, 226, 13)
    # elif _cls == 40:                # obstacle_bollard_cylinder
    #     color = (52, 226, 13)
    # elif _cls == 41:                # obstacle_bollard_marker
    #     color = (79, 233, 252)
    # elif _cls == 42:                # obstacle_bollard_stone
    #     color = (41, 41, 239)
    # elif _cls == 43:                # obstacle_bollard_U_shaped
    #     color = (17, 125, 193)
    # elif _cls == 44:                # obstacle_bollard_barricade
    #     color = (207, 159, 114)
    # elif _cls == 45:                # obstacle_cylinder
    #     color = (79, 233, 252)
    # elif _cls == 46:                # parking_sign
    #     color = (255, 153, 51)
    # elif _cls == 47:                # parking_lock
    #     color = (0, 128, 255)
    # elif _cls == 48:                # parking_cylinder
    #     color = (52, 226, 13)
    # elif _cls == 49:                # parking_stopper_marble
    #     color = (52, 226, 13)
    # elif _cls == 50:                # parking_stopper_bar
    #     color = (52, 226, 13)
    # elif _cls == 51:                # parking_stopper_separated
    #     color = (52, 226, 13)
    

    return color



def find_feature(feature_list, fname):
    for f in feature_list:
        if f == fname:
            return True
    return False



if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    model_type = args.model_type
    model_type = str(model_type).upper()

    metadata_ffc = [        
                'pedestrian',
		        'sit_person', 		
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
                'ignored'
            ]
    
    metadata_svm = [        
                'pedestrian',
		        'car', 		
		        'truck', 
                'bus', 
                '3-wheels',
		        'excavator',
                'forklift',
                'rider_bicycle',
                'rider_bike',
                'trailer',
                'van',
                'obstacle_cone',
                'obstacle_cylinder',
                'obstacle_drum',
                'obstacle_bollard_cylinder',
                'obstacle_bollard_stone',
                'obstacle_bollard_U_shaped',
                'obstacle_bollard_barricade', 
                'obstacle_bollard_marker',
                'obstacle_bollard_special',
                'parking_cylinder',
                'parking_sign',
                'parking_stopper_separated',
                'parking_stopper_bar',
                'parking_stopper_marble',
                'blocking_bar',
                'ignored'
            ]

    metadata_svm_od = [        
                'pedestrian',
		        'car',
		        'truck',
                'bus',
                '3-wheels',
		        'rider_bicycle',
                'rider_bike',
                'ignored'
            ]

    nBatch = 32

    if model_type == 'FFC':
        output_prefix = 'ALT'
        metadata = metadata_ffc
    elif model_type == 'SVM':
        nBatch = 24
        output_prefix = 'ALT.SVM'
        metadata = metadata_svm
    else:
        sys.exit()

    
    args.opts.append('MODEL.ROI_HEADS.NMS_THRESH_TEST')
    args.opts.append(1.0)
    # args.opts.append('MODEL.ROI_HEADS.SCORE_THRESH_TEST')
    # args.opts.append(0.1)
    
    cfg = setup_cfg(args)
    
    model = Trainer.build_model(cfg)

    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    if cfg.TEST.AUG.ENABLED:
        model = GeneralizedRCNNWithTTA(cfg, model, batch_size=nBatch)
    model.eval()
    
    transform_gen = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
        
    args.input = glob.glob(os.path.expanduser(args.input[0]))
    # args.input = glob.glob(glob.escape(os.path.expanduser(args.input[0])))

    img_list = os.listdir(args.input[0])
    img_list = [file for file in img_list if 'jpg' in str(file).lower() or 'png' in str(file).lower()]
    img_list.sort()
    nImgs = len(img_list)


    args.input_xml = glob.glob(os.path.expanduser(args.input_xml[0]))
    xml_list = os.listdir(args.input_xml[0])
    xml_list = [file for file in xml_list if 'xml' in str(file).lower()]
    xml_list.sort()

    if not os.path.isdir(args.output):
        os.makedirs(args.output)
        os.makedirs(os.path.join(args.output, output_prefix + '.OUT_XML'))
        if args.save_outimg:
            os.makedirs(os.path.join(args.output, output_prefix + '.OUT_IMG'))            
    else:
        if not os.path.isdir(os.path.join(args.output, output_prefix + '.OUT_XML')):
            os.makedirs(os.path.join(args.output, output_prefix + '.OUT_XML'))
        if args.save_outimg:
            if not os.path.isdir(os.path.join(args.output, output_prefix + '.OUT_IMG')):
                os.makedirs(os.path.join(args.output, output_prefix + '.OUT_IMG'))
                    

    batch_list = [img_list[i:i + nBatch] for i in range(0, len(img_list), nBatch)]
    xml_batch_list = [xml_list[i:i + nBatch] for i in range(0, len(xml_list), nBatch)]

    target_w = cfg.INPUT.MAX_SIZE_TEST
    target_h = math.trunc(800 / 1333 * target_w)

    count = 0
    with torch.no_grad():
        for idx, batch in enumerate(batch_list):            
            batch = [os.path.join(args.input[0], i) for i in batch]
            
            xml_batch = xml_batch_list[idx]
            xml_batch = [os.path.join(args.input_xml[0], i) for i in xml_batch]

            img_sizes = []
            inputs = []

            for idx, b in enumerate(batch):
                # xml
                xml = load_udb_annotation(xml_batch[idx])

                # if len(xml) != 0:
                #     print('debug')

                image = read_image(b, format="BGR")
                height, width = image.shape[:2]
                img_sizes.append([height, width, 3])

                gt_boxes = [obj["gt_bbox"] for obj in xml]
                
                gt_class_ids = [obj["category_id"] for obj in xml]

                # gt scaling: x' = x / width * target_w, y' = y / height * target_h
                gt_boxes = [[box[0] / width * target_w, box[1] / height * target_h, box[2] / width * target_w, box[3] / height * target_h] for box in gt_boxes]

                xml_input = Instances(image.shape[:2])
                xml_input.proposal_boxes = Boxes(torch.as_tensor(gt_boxes))
                # xml_input.proposal_boxes = torch.as_tensor(gt_boxes, device=torch.device("cuda"))

                if cfg.TEST.AUG.ENABLED:
                    # input = {"file_name": b, "height": height, "width": width}
                    img = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
                    input = {"image": img, "height": height, "width": width}
                else:
                    img = transform_gen.get_transform(image).apply_image(image)
                    img = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
                    if cfg.MODEL.USE_CLASSIFIER_ONLY:
                        input = {"image": img, "height": height, "width": width, "proposals": xml_input, "gt_class_id": gt_class_ids}
                    else:
                        input = {"image": img, "height": height, "width": width}
                    
                
                input["filepath"] = b
                inputs.append(input)
            
            result = model(inputs)      # 0: detection, 1: proposal


            if cfg.TEST.AUG.ENABLED:
                result = [result]


            for i in range(len(result[0])):
                h = inputs[i]["height"]
                w = inputs[i]["width"]

                detection = []

                for j in range(len(result[0][i]['instances'])):
                    bbox = result[0][i]['instances'][j].pred_boxes.tensor.cpu().numpy()
                    # gt scaling: x' = x / width * target_w, y' = y / height * target_h
                    xmin = bbox[0][0] / target_w * img_sizes[i][1]
                    ymin = bbox[0][1] / target_h * img_sizes[i][0]
                    xmax = bbox[0][2] / target_w * img_sizes[i][1]
                    ymax = bbox[0][3] / target_h * img_sizes[i][0]
                    # xmin = bbox[0][0] * w / img_sizes[i][1]
                    # ymin = bbox[0][1] * h / img_sizes[i][0]
                    # xmax = bbox[0][2] * w / img_sizes[i][1]
                    # ymax = bbox[0][3] * h / img_sizes[i][0]                    
                    dcls = result[0][i]['instances'][j].pred_classes.tolist()[0]
                    
                    ## xml
                    if model_type == 'FFC':
                        if metadata_ffc[inputs[i]['gt_class_id'][j]] == 'ignored':
                            detection.append({"bbox": [xmin, ymin, xmax, ymax], "class_name": 'ignored_' + metadata_ffc[dcls]})
                        else:
                            detection.append({"bbox": [xmin, ymin, xmax, ymax], "class_name": metadata_ffc[dcls]})
                    elif model_type == 'SVM':
                        if metadata_svm[inputs[i]['gt_class_id'][j]] == 'ignored':
                            detection.append({"bbox": [xmin, ymin, xmax, ymax], "class_name": 'ignored_' + metadata_svm[dcls]})
                        else:
                            detection.append({"bbox": [xmin, ymin, xmax, ymax], "class_name": metadata_svm[dcls]})
                
                # output path
                output_xml = os.path.join(os.path.join(args.output, output_prefix + '.OUT_XML'), os.path.basename(xml_batch[i]))
                write_xml2(img_sizes[i], detection, output_xml)


                if args.save_outimg == 1:
                    filename = os.path.basename(os.path.splitext(batch[i])[0])

                    outimgfilepath = os.path.join(args.output, output_prefix + '.OUT_IMG')
                    outimgfilepath = os.path.join(outimgfilepath, filename + '.jpg')

                    image = cv2.imread(batch[i])
                   
                    # new
                    for obj in detection:
                        bbox = obj["bbox"]
                        xmin = int(bbox[0])
                        ymin = int(bbox[1])
                        xmax = int(bbox[2])
                        ymax = int(bbox[3])

                        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)                    
                        cv2.putText(image, obj["class_name"], (int(xmin), int(ymin)-5), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
                        cv2.putText(image, obj["class_name"], (int(xmin), int(ymin)-5), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

                    cv2.imwrite(outimgfilepath, image)
                            
            count = count + len(result[0])          
            
            print('[{}/{}]'.format(count, nImgs))
    print('All inferences have been finished')
    
