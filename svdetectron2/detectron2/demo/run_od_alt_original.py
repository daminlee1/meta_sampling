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

from tools.train_net import Trainer
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import GeneralizedRCNNWithTTA
from collections import OrderedDict
import detectron2.data.transforms as T
from detectron2.data import MetadataCatalog
import torch

from detectron2.utils.inference_util import write_xml

import xml.etree.ElementTree as ET
from detectron2.data.datasets import udb_datasets 



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
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--low-confidence-threshold",
        type=float,
        default=0.5,
        help="low confidence threshold for ignoring object",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )    
    parser.add_argument("--proposal-iou", action="store_true", help="extract proposal iou distribution.")
    parser.add_argument("--save-outimg", action="store_true", help="save output image.")
    parser.add_argument("--save-outtxt", action="store_true", help="save output textfile.")
    
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


def match_gt_detection(gt, detection, conf_th = 0.9):
    # det_boxes = [det['instances'].pred_boxes.tensor.cpu().numpy() for det in detection]
    # det_scores = [det['instances'].scores.cpu().numpy() for det in detection]
    # det_cls = [det['instances'].pred_classes.cpu().numpy() for det in detection]
    
    # # filtering by score
    # det_boxes = [boxes[det_scores[idx] > conf_th] for idx, boxes in enumerate(det_boxes)]
    # det_cls = [dcls[det_scores[idx] > conf_th] for idx, dcls in enumerate(det_cls)]

    # unmatched = []
    # for i in range(len(gt)):    # batch level        
    #     check_obj = {}
    #     unmatched_obj = []
    #     miscls_obj  = []

    #     for idx, gtobj in enumerate(gt[i]):
    #         if gtobj['category_id'] == 9:
    #             continue
    #         iou, iou_max, nmax = get_max_iou(det_boxes[i], gtobj['gt_bbox'])
            
    #         if iou_max < 0.5:
    #             unmatched_obj.append(idx)
    #         elif iou_max >= 0.5 and gtobj['category_id'] != det_cls[i][nmax]:
    #             miscls_obj.append(idx)
    #     check_obj["unmatched_obj"] = unmatched_obj
    #     check_obj["miscls_obj"] = miscls_obj
    #     unmatched.append(check_obj)

    # return unmatched


    det_boxes = detection['instances'].pred_boxes.tensor.cpu().numpy()
    det_scores = detection['instances'].scores.cpu().numpy()
    det_cls = detection['instances'].pred_classes.cpu().numpy()

    # filtering by score
    det_boxes = det_boxes[det_scores > conf_th]
    det_cls = det_cls[det_scores > conf_th]

        
    check_obj = {}
    unmatched_obj = []
    miscls_obj  = []

    if det_boxes.shape[0] != 0:
        for idx, gtobj in enumerate(gt):
            if gtobj['category_id'] == 9:
                continue
            iou, iou_max, nmax = get_max_iou(det_boxes, gtobj['gt_bbox'])
                
            if iou_max < 0.5:
                unmatched_obj.append(idx)
            elif iou_max >= 0.5 and gtobj['category_id'] != det_cls[nmax]:
                miscls_obj.append(idx)
    check_obj["unmatched_obj"] = unmatched_obj
    check_obj["miscls_obj"] = miscls_obj

    return check_obj



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



if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    nBatch = 12
    cfg = setup_cfg(args)
    
    model = Trainer.build_model(cfg)

    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    if cfg.TEST.AUG.ENABLED:
        model = GeneralizedRCNNWithTTA(cfg, model, batch_size=nBatch)
    model.eval()
  
    metadata = [        
        'Ped',
		'Sit_Person', 		
		'Rider_Bike', 
        'Rider_Bicycle', 
        'Bicycle', 
		'Bike',		
		'3-Wheels',
        '3-Wheels_Rider',		
        'Sedan', 
        'Van',         
        'Truck',        
		'Box_Truck', 
		'Pickup_Truck',		
		'Mixer_Truck',
		'Ladder_Truck', 
        'Bus',		
		'Trailer',                 
		'Excavator',
		'Forklift',		
		'TS_Triangle',
		'TS_Inv_Triangle', 		
		'TS_Cir',
		'TS_Cir_Speed', 
		'TS_Rect', 
        'TS_Rect_Speed', 
		'TS_Rect_Arrow', 
        'TS_Diamonds',         		
        'TS_Rear',
		'TS_Main_Zone', 
		'TS_Sup_Letter',
        'TS_Sup_Arrow',
		'TS_Sup_Zone',
		'TS_Sup_Drawing', 		        
        'TL_Car',
		'TL_Ped',
        'TL_Rear',
		'TL_Light_Only', 	
        'Blk_Bar', 		
        'Obs_Drum', 
        'Obs_Cone', 
        'Obs_Bollard_Cylinder', 
		'Obs_Bollard_Marker',
		'Obs_Bollard_Stone',
		'Obs_Bollard_U_Shaped',
		'Obs_Bollard_Barricade', 
        'Obs_Cylinder',
        'PK_Sign',   
		'PK_Lock', 		
        'PK_Cylinder',
        'PK_Stopper_Marble', 
        'PK_Stopper_Bar',
        'PK_Stopper_Separated',
        'Ignored'
    ]
    
    transform_gen = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
    
    
    args.input = glob.glob(os.path.expanduser(args.input[0]))
    img_list = os.listdir(args.input[0])
    nImgs = len(img_list)

     

    if not os.path.isdir(args.output):
        os.makedirs(args.output)
        # os.makedirs(os.path.join(args.output, 'out_xml'))
        if args.save_outimg:
            os.makedirs(os.path.join(args.output, 'out_img'))
            # os.makedirs(os.path.join(args.output, 'out_img_err'))
    else:
        # if not os.path.isdir(os.path.join(args.output, 'out_xml')):
        #     os.makedirs(os.path.join(args.output, 'out_xml'))
        # if args.save_outimg:
            if not os.path.isdir(os.path.join(args.output, 'out_img')):
                os.makedirs(os.path.join(args.output, 'out_img'))
            # if not os.path.isdir(os.path.join(args.output, 'out_img_err')):
                # os.makedirs(os.path.join(args.output, 'out_img_err'))
        

    batch_list = [img_list[i:i + nBatch] for i in range(0, len(img_list), nBatch)]
    
    unmatched_imgfiles = []
    unmatched_boxes = []

    count = 0
    with torch.no_grad():
        for idx, batch in enumerate(batch_list):            
            batch = [os.path.join(args.input[0], i) for i in batch]            
            
            inputs = []            
            for idx, b in enumerate(batch):
                image = read_image(b, format="BGR")
                height, width = image.shape[:2]

                if width == 1280 and height == 960:
                    image = image[0:720, 0:1280]
                    height, width = image.shape[:2]
                    
                if cfg.TEST.AUG.ENABLED:
                    # input = {"file_name": b, "height": height, "width": width}
                    img = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
                    input = {"image": img, "height": height, "width": width}
                else:
                    img = transform_gen.get_transform(image).apply_image(image)
                    img = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
                    input = {"image": img, "height": height, "width": width}
                
                input["filepath"] = b                
                inputs.append(input)
            
            result = model(inputs)      # 0: detection, 1: proposal


            if cfg.TEST.AUG.ENABLED:
                result = [result]


            ## proposals & detections
            for i in range(len(result[0])):
                h = inputs[i]["height"]
                w = inputs[i]["width"]
                
                for j in range(len(result[0][i]['instances'])):
                    bbox = result[0][i]['instances'][j].pred_boxes.tensor.cpu().numpy()
                    # cx = (bbox[0][2] - bbox[0][0] + 1) * 1280 / w
                    # cy = bbox[0][3] - bbox[0][1] + 1

                    xmin = bbox[0][0] * w / 1280
                    ymin = bbox[0][1] * h / 720
                    xmax = bbox[0][0] * w / 1280
                    ymax = bbox[0][1] * h / 720
                    score = result[0][i]['instances'][j].scores.cpu().numpy()[0]
                    dcls = result[0][i]['instances'][j].pred_classes.tolist()[0] + 1
                    
                filename = os.path.basename(os.path.splitext(batch[i])[0])

                outxmlfilepath = os.path.join(args.output, 'out_xml')
                outxmlfilepath = os.path.join(outxmlfilepath, filename + '.xml')
                # write_xml([inputs[i]["height"], inputs[i]["width"], 3], result[0][i], metadata, outxmlfilepath)

                if args.save_outtxt:
                    od_strs = []
                    tstld_strs = []
                    sod_strs = []                    

                    for j in range(len(result[0][i]['instances'])):
                        bbox = result[0][i]['instances'][j].pred_boxes.tensor.cpu().numpy()
                        xmin = bbox[0][0] * w / 1280
                        ymin = bbox[0][1] * h / 720
                        xmax = bbox[0][0] * w / 1280
                        ymax = bbox[0][1] * h / 720
                        score = result[0][i]['instances'][j].scores.cpu().numpy()[0]
                        dcls = result[0][i]['instances'][j].pred_classes.tolist()[0]
                        adj_cls = dcls

                        if dcls >= 0 and dcls <= 18:
                            adj_cls = dcls
                        elif dcls >= 19 and dcls <= 36:
                            adj_cls = dcls - 19
                        elif dcls >= 37 and dcls <= 51:
                            adj_cls = dcls - 37

                        obj_str = str(adj_cls) + ' ' + str(format(xmin, ".2f")) + ' ' + str(format(ymin, ".2f")) + ' ' + str(format(xmax, ".2f")) + ' ' + str(format(ymax, ".2f")) + ' ' + '""'
                        
                        if dcls >= 0 and dcls <= 18:
                            od_strs.append(obj_str)
                        elif dcls >= 19 and dcls <= 36:
                            tstld_strs.append(obj_str)
                        elif dcls >= 37 and dcls <= 51:
                            sod_strs.append(obj_str)


                    with open(os.path.join(args.output, 'labels_od.txt'), 'a') as f:
                        f.write(os.path.basename(batch[i]))
                        f.write('\n')
                        f.write(str(len(od_strs)))
                        f.write('\n')
                        f.writelines("%s\n" % item for item in od_strs)                        

                    with open(os.path.join(args.output, 'labels_tstld.txt'), 'a') as f:
                        f.write(os.path.basename(batch[i]))
                        f.write('\n')
                        f.write(str(len(tstld_strs)))
                        f.write('\n')
                        f.writelines("%s\n" % item for item in tstld_strs)                        

                    with open(os.path.join(args.output, 'labels_sod.txt'), 'a') as f:
                        f.write(os.path.basename(batch[i]))
                        f.write('\n')
                        f.write(str(len(sod_strs)))
                        f.write('\n')
                        f.writelines("%s\n" % item for item in sod_strs)
                        

                if args.save_outimg:
                    outimgfilepath = os.path.join(args.output, 'out_img')
                    outimgfilepath = os.path.join(outimgfilepath, filename + '.jpg')

                    # image = inputs[i]["image"].permute(1, 2, 0).numpy().astype("uint8")[:, :, ::-1].copy()
                    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    image = cv2.imread(batch[i])

                    # visualizer = Visualizer(image, metadata, instance_mode=ColorMode.IMAGE)
                    # if "instances" in result[0][i]:
                    #     instances = result[0][i]["instances"].to(torch.device("cpu"))
                    #     vis_output = visualizer.draw_instance_predictions(predictions=instances)
                    #     vis_output.save(outimgfilepath)

                    ## draw gt and propossals
                    for j in range(len(result[0][i]['instances'])):
                        bbox = result[0][i]['instances'][j].pred_boxes.tensor.cpu().numpy()
                        # bbox = result[1][i]['instances'][j].proposal_boxes.tensor.cpu().numpy()
                        xmin = bbox[0][0]
                        ymin = bbox[0][1]
                        xmax = bbox[0][2]
                        ymax = bbox[0][3]
                        score = result[0][i]['instances'][j].scores.cpu().numpy()[0]
                        dcls = result[0][i]['instances'][j].pred_classes.tolist()[0]
                                                
                        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)                    
                        cv2.putText(image, metadata[dcls], (int(xmin), int(ymin)-5), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
                        cv2.putText(image, metadata[dcls], (int(xmin), int(ymin)-5), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imwrite(outimgfilepath, image)
                
            count = count + len(result[0])          
            
            print('[{}/{}]'.format(count, nImgs))
    print('All inferences have been finished')
    