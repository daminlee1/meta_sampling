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

from evaluations.detectionbox import DetectionBox
import pickle
import _pickle
import numpy as np



# constants
WINDOW_NAME = "COCO detections"


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
    # parser.add_argument(
    #     "--pkl_append",
    #     help="postfix naming"        
    # )
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
    return parser


def get_iou(pred_box, gt_box):
    """
    pred_box : the coordinate for predict bounding box
    gt_box :   the coordinate for ground truth bounding box
    return :   the iou score
    the  left-down coordinate of  pred_box:(pred_box[0], pred_box[1])
    the  right-up coordinate of  pred_box:(pred_box[2], pred_box[3])
    """
    # 1.get the coordinate of inters
    ixmin = max(pred_box[0], gt_box[0])
    ixmax = min(pred_box[2], gt_box[2])
    iymin = max(pred_box[1], gt_box[1])
    iymax = min(pred_box[3], gt_box[3])

    iw = np.maximum(ixmax-ixmin+1., 0.)
    ih = np.maximum(iymax-iymin+1., 0.)

    # 2. calculate the area of inters
    inters = iw*ih

    # 3. calculate the area of union
    uni = ((pred_box[2]-pred_box[0]+1.) * (pred_box[3]-pred_box[1]+1.) +
           (gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.) -
           inters)

    # 4. calculate the overlaps between pred_box and gt_box
    iou = inters / uni

    return iou


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


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)


    args.input = glob.glob(os.path.expanduser(args.input[0]))
        
    with open(os.path.join(args.input[0], 'ImageSets/all.txt')) as f:
        img_list = [line.rstrip() + '.jpg' for line in f]

    
    detection_boxes = []
    for _ in img_list:
        detection_boxes.append([])       

    proposal_boxes =  [ None for _ in img_list ]
    

    for i in range(len(img_list)):      
        path = os.path.join(args.input[0], 'JPEGImages')  
        path = os.path.join(path, img_list[i])
        img = read_image(path, format="BGR")
        predictions, visualized_output = demo.run_on_image(img)
        
        # im2 = cv2.imread(path)

        print('inference on ' + str(i) + ' image')

        proposal_boxes[i] = []
        # proposal_boxes[i].append([])
        for j in range(len(predictions[1])):
            proposal_box = predictions[1][0]['instances'].proposal_boxes.tensor.cpu().numpy()
            proposal_boxes[i].append(proposal_box)

            # # small_proposals= proposal_box[proposal_box[:,3] - proposal_box[:,1] < 32, :]
            # small_proposals= proposal_box[np.logical_and(proposal_box[:,3] - proposal_box[:,1] >= 32, proposal_box[:,3] - proposal_box[:,1] < 64), :]
            
            # if len(small_proposals) > 0:
            #     for k in range(len(small_proposals)):
            #         # iou calculation
            #         cv2.rectangle(im2, (small_proposals[k][0], small_proposals[k][1]), (small_proposals[k][2], small_proposals[k][3]), (0, 255, 0), 1)


        small_objs = []
        iou_th = 0.0

        for j in range(len(predictions[0][0]['instances'])):
            bbox = predictions[0][0]['instances'][j].pred_boxes.tensor.cpu().numpy()
            xmin = bbox[0][0]
            ymin = bbox[0][1]
            xmax = bbox[0][2]
            ymax = bbox[0][3]
            score = predictions[0][0]['instances'][j].scores.cpu().numpy()[0]
            dcls = predictions[0][0]['instances'][j].pred_classes.tolist()[0] + 1
            detection_boxes[i].append(DetectionBox(int(dcls), xmin, ymin, xmax, ymax, score))

        #     w = xmax - xmin + 1
        #     h = ymax - ymin + 1
        #     if h <= 32:
        #         small_objs.append([xmin, ymin, xmax, ymax])
        #         cv2.rectangle(im2, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
                
        #         # calculate iou
        #         ious, max_iou, nmax = get_max_iou(proposal_boxes[i][0], bbox[0])
              
        #         indices = np.where(ious > iou_th)[0]
        #         for k in range(len(indices)):
        #             pbox = proposal_boxes[i][0][k]
        #             cv2.rectangle(im2, (pbox[0], pbox[1]), (pbox[2], pbox[3]), (0, 255, 0), 1)

        # cv2.imwrite('result_test.jpg', im2)

    print('inference was finished!')

    db_name = os.path.basename(args.input[0])

    rpn_topk = cfg.MODEL.RPN.POST_NMS_TOPK_TEST

    det_file = os.path.join(args.output[0], db_name + "_" + str(rpn_topk) + "_detections.pkl")
    with open(det_file, "wb") as f:
        _pickle.dump(detection_boxes, f, protocol=2)

    rpn_file = os.path.join(args.output[0], db_name + "_" + str(rpn_topk) + "_proposals.pkl")
    with open(rpn_file, "wb") as f:
        _pickle.dump(proposal_boxes, f, protocol=2)

    print('All finished')
    