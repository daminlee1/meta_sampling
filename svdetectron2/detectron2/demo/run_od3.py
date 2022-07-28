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
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )    
    parser.add_argument("--proposal-iou", action="store_true", help="extract proposal iou distribution.")
    parser.add_argument("--save-outimg", action="store_true", help="save output image.")
    
    return parser


def load_udb_annotation(xml_file):
    cls_dict = udb_datasets.get_god_2020_10cls_dataset()

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
        if cls_dict.cls_map[cls] - 1 != 9:  # etc class_id = 9
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


def match_gt_detection(gt, detection, conf_th = 0.9, iou_th = 0.5):
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
    imprecise_obj = []
   
    miscls_det_obj  = []
    imprecise_det_obj = []
        
    
    if det_boxes.shape[0] != 0:
        det_boxes_tmp = det_boxes.copy()
        for idx, gtobj in enumerate(gt):
            if gtobj['category_id'] == 9:
                continue
            iou, iou_max, nmax = get_max_iou(det_boxes_tmp, gtobj['gt_bbox'])
            det_boxes_tmp[nmax] = [0.0, 0.0, 0.2, 0.2]
            
            if iou_max >= iou_th and gtobj['category_id'] != det_cls[nmax]:
                miscls_obj.append(idx)
                miscls_det_obj.append(nmax)
            elif iou_max < iou_th and gtobj['category_id'] == det_cls[nmax]:
                imprecise_obj.append(idx)
                imprecise_det_obj.append(nmax)
            elif iou_max < iou_th:
                unmatched_obj.append(idx)
    check_obj["unmatched_obj"] = unmatched_obj
    check_obj["miscls_obj"] = miscls_obj
    ###
    check_obj["imprecise_obj"] = imprecise_obj
    check_obj["miscls_det_obj"] = miscls_det_obj
    check_obj["imprecise_det_obj"] = imprecise_det_obj   
    ###

    return check_obj



def get_color(_cls):
    color = (0, 0, 0)   # bg
    if _cls == 0:
        color = (52, 226, 138)
    elif _cls == 1:
        color = (79, 233, 252)
    elif _cls == 2:
        color = (41, 41, 239)
    elif _cls == 3:
        color = (17, 125, 193)
    elif _cls == 4:
        color = (207, 159, 114)
    elif _cls == 5: # ts_t
        color = (79, 233, 252)
    elif _cls == 6: # ts_c
        color = (255, 153, 51)
    elif _cls == 7: # ts_r
        color = (0, 128, 255)
    elif _cls == 8: # tl
        color = (52, 226, 13)

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
  
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
    
    transform_gen = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
    
    
    args.input = glob.glob(os.path.expanduser(args.input[0]))
    img_list = os.listdir(args.input[0])
    nImgs = len(img_list)

    
    ###############################
    with open(os.path.join(args.input[0], 'ImageSets/all.txt')) as f:
        img_list = [line.rstrip() + '.jpg' for line in f]
        nImgs = len(img_list)

    anno_list = [imf.replace('jpg', 'xml') for imf in img_list]

    detection_boxes = []
    for _ in img_list:
        detection_boxes.append([])


    proposal_boxes =  [ None for _ in img_list ]

    detection_errfile_list = []
    ###############################

    if not os.path.isdir(args.output):
        os.makedirs(args.output)
        os.makedirs(os.path.join(args.output, 'out_xml'))
        if args.save_outimg:
            os.makedirs(os.path.join(args.output, 'out_img'))
            os.makedirs(os.path.join(args.output, 'out_img_err'))
    else:
        if not os.path.isdir(os.path.join(args.output, 'out_xml')):
            os.makedirs(os.path.join(args.output, 'out_xml'))
        if args.save_outimg:
            if not os.path.isdir(os.path.join(args.output, 'out_img')):
                os.makedirs(os.path.join(args.output, 'out_img'))
            if not os.path.isdir(os.path.join(args.output, 'out_img_err')):
                os.makedirs(os.path.join(args.output, 'out_img_err'))
        

    batch_list = [img_list[i:i + nBatch] for i in range(0, len(img_list), nBatch)]
    
    unmatched_imgfiles = []
    unmatched_boxes = []

    count = 0
    with torch.no_grad():
        for idx, batch in enumerate(batch_list):            
            batch = [os.path.join(args.input[0], 'JPEGImages/' + i) for i in batch]
            # batch = [os.path.join(args.input[0], i) for i in batch]

            # gt
            batch_anno = [b.replace("JPEGImages", "Annotations").replace("jpg", "xml") for b in batch]

            inputs = []
            for idx, b in enumerate(batch):
                image = read_image(b, format="BGR")
                height, width = image.shape[:2]

                if cfg.TEST.AUG.ENABLED:
                    # input = {"file_name": b, "height": height, "width": width}
                    img = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
                    input = {"image": img, "height": height, "width": width}
                else:
                    img = transform_gen.get_transform(image).apply_image(image)
                    img = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
                    input = {"image": img, "height": height, "width": width}

                # load gt
                anno = load_udb_annotation(batch_anno[idx])
                input["annotations"] = anno
                input["filepath"] = b
                
                inputs.append(input)
            
            result = model(inputs)      # 0: detection, 1: proposal


            if cfg.TEST.AUG.ENABLED:
                result = [result]       


            ## proposals & detections
            for i in range(len(result[0])):
                proposal_boxes[count + i] = []
                proposal_box = result[1][i]['instances'].proposal_boxes.tensor.cpu().numpy()
                proposal_boxes[count + i].append(proposal_box)                    


                for j in range(len(result[0][i]['instances'])):
                    bbox = result[0][i]['instances'][j].pred_boxes.tensor.cpu().numpy()
                    xmin = bbox[0][0]
                    ymin = bbox[0][1]
                    xmax = bbox[0][2]
                    ymax = bbox[0][3]
                    score = result[0][i]['instances'][j].scores.cpu().numpy()[0]
                    dcls = result[0][i]['instances'][j].pred_classes.tolist()[0] + 1
                    detection_boxes[count + i].append(DetectionBox(int(dcls), xmin, ymin, xmax, ymax, score))


                filename = os.path.basename(os.path.splitext(batch[i])[0])

                # outxmlfilepath = os.path.join(args.output, 'out_xml')
                # outxmlfilepath = os.path.join(outxmlfilepath, filename + '.xml')
                # write_xml([inputs[i]["height"], inputs[i]["width"], 3], result[0][i], metadata.thing_classes, outxmlfilepath)


                if args.save_outimg:
                    outimgfilepath = os.path.join(args.output, 'out_img')
                    outimgfilepath = os.path.join(outimgfilepath, filename + '.jpg')

                    image = inputs[i]["image"].permute(1, 2, 0).numpy().astype("uint8")[:, :, ::-1].copy()
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    # visualizer = Visualizer(image, metadata, instance_mode=ColorMode.IMAGE)
                    # if "instances" in result[0][i]:
                    #     instances = result[0][i]["instances"].to(torch.device("cpu"))
                    #     vis_output = visualizer.draw_instance_predictions(predictions=instances)
                    #     vis_output.save(outimgfilepath)

                    ## draw gt and propossals
                    for anno in inputs[i]['annotations']:
                        bbox = anno['gt_bbox']
                        cv2.rectangle(image, (np.int(bbox[0]), np.int(bbox[1])), (np.int(bbox[2]), np.int(bbox[3])), (255, 255, 255), 2)    # white
                    # for pbox in proposal_box:
                    #     cv2.rectangle(image, (pbox[0], pbox[1]), (pbox[2], pbox[3]), (0, 0, 255), 1)
                    for j in range(len(result[0][i]['instances'])):
                        bbox = result[0][i]['instances'][j].pred_boxes.tensor.cpu().numpy()
                        # bbox = result[1][i]['instances'][j].proposal_boxes.tensor.cpu().numpy()
                        xmin = bbox[0][0]
                        ymin = bbox[0][1]
                        xmax = bbox[0][2]
                        ymax = bbox[0][3]
                        score = result[0][i]['instances'][j].scores.cpu().numpy()[0]
                        dcls = result[0][i]['instances'][j].pred_classes.tolist()[0]
                                                
                        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), get_color(dcls), 1)                    
                    cv2.imwrite(outimgfilepath, image)



                    ########## matching with gt
                    # gt = [_input['annotations'] for _input in inputs]
                    # det_errs = match_gt_detection(gt, result[0])
                    det_errs = match_gt_detection(inputs[i]['annotations'], result[0][i], iou_th = 0.8)
                    
                    # visualization error
                    if 1:
                        image = inputs[i]["image"].permute(1, 2, 0).numpy().astype("uint8")[:, :, ::-1].copy()
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                        outimgfilepath = os.path.join(args.output, 'out_img_err')
                        outimgfilepath = os.path.join(outimgfilepath, filename + '.jpg')

                        # for idx in det_errs['unmatched_obj']:
                        #     bbox = inputs[i]['annotations'][idx]['gt_bbox']
                        #     cv2.rectangle(image, (np.int(bbox[0]), np.int(bbox[1])), (np.int(bbox[2]), np.int(bbox[3])), (255, 255, 255), 1)

                        nMiscls = len(det_errs['miscls_obj'])
                        for idx in det_errs['miscls_obj']:
                            bbox = inputs[i]['annotations'][idx]['gt_bbox']
                            gtcls = inputs[i]['annotations'][idx]['category_id']
                            if gtcls == -1:
                                nMiscls = nMiscls - 1
                                continue
                            cv2.rectangle(image, (np.int(bbox[0]), np.int(bbox[1])), (np.int(bbox[2]), np.int(bbox[3])), get_color(gtcls), 2)

                        nImprecise = len(det_errs['imprecise_obj'])
                        for idx in det_errs['imprecise_obj']:
                            bbox = inputs[i]['annotations'][idx]['gt_bbox']
                            gtcls = inputs[i]['annotations'][idx]['category_id']
                            if gtcls == -1:
                                nImprecise = nImprecise - 1
                                continue
                            cv2.rectangle(image, (np.int(bbox[0]), np.int(bbox[1])), (np.int(bbox[2]), np.int(bbox[3])), (255, 255, 255), 2)
                        

                        ###############
                        det_boxes = result[0][i]['instances'].pred_boxes.tensor.cpu().numpy()
                        det_classes = result[0][i]['instances'].pred_classes.cpu().numpy()

                        for idx in det_errs['miscls_det_obj']:
                            # bbox = result[0][i]['instances'][idx].pred_boxes.tensor.cpu().numpy()
                            # detcls = result[0][i]['instances'][idx].pred_classes.cpu().numpy()
                            bbox = det_boxes[idx]
                            detcls = det_classes[idx]
                            cv2.rectangle(image, (np.int(bbox[0]), np.int(bbox[1])), (np.int(bbox[2]), np.int(bbox[3])), get_color(detcls), 1)
                        

                        for idx in det_errs['imprecise_det_obj']:                            
                            # bbox = result[0][i]['instances'][idx].pred_boxes.tensor.cpu().numpy()
                            # detcls = result[0][i]['instances'][idx].pred_classes.cpu().numpy()
                            bbox = det_boxes[idx]
                            detcls = det_classes[idx]
                            cv2.rectangle(image, (np.int(bbox[0]), np.int(bbox[1])), (np.int(bbox[2]), np.int(bbox[3])), (0, 0, 0), 1)
                        

                        # if len(det_errs['unmatched_obj']) != 0 or len(det_errs['miscls_obj']) != 0:
                        #     detection_errfile_list.append(inputs[i]["filepath"])

                        if nMiscls != 0 or nImprecise != 0:
                            detection_errfile_list.append(inputs[i]["filepath"] + '\n')
                            cv2.imwrite(outimgfilepath, image)


                    ##########
                
            count = count + len(result[0])          
            
            print('[{}/{}]'.format(count, nImgs))


    # dump result
    db_name = os.path.basename(args.input[0])

    rpn_topk = cfg.MODEL.RPN.POST_NMS_TOPK_TEST

    if cfg.TEST.AUG.ENABLED:
        det_file = os.path.join(args.output, db_name + "_" + str(rpn_topk) + "_tta_detections.pkl")
    else:
        det_file = os.path.join(args.output, db_name + "_" + str(rpn_topk) + "_detections.pkl")
    with open(det_file, "wb") as f:
        _pickle.dump(detection_boxes, f, protocol=2)

    rpn_file = os.path.join(args.output, db_name + "_" + str(rpn_topk) + "_proposals.pkl")
    with open(rpn_file, "wb") as f:
        _pickle.dump(proposal_boxes, f, protocol=2)

    det_err_file = os.path.join(args.output, db_name + "_" + "_detection_error.txt")
    with open(det_err_file, "w") as f:
        f.writelines(detection_errfile_list)
                

    print('All inferences have been finished')
    