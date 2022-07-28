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
import pickle
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
    parser.add_argument(
        "--save-outimg",
        help="save output image",
        default=False,        
    )
    return parser



if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    nBatch = 24
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


    if not os.path.isdir(args.output):
        os.makedirs(args.output)
        os.makedirs(os.path.join(args.output, 'out_xml'))
        if args.save_outimg:
            os.makedirs(os.path.join(args.output, 'out_img'))
    else:
        if not os.path.isdir(os.path.join(args.output, 'out_xml')):
            os.makedirs(os.path.join(args.output, 'out_xml'))
        if args.save_outimg:
            if not os.path.isdir(os.path.join(args.output, 'out_img')):
                os.makedirs(os.path.join(args.output, 'out_img'))
        

    batch_list = [img_list[i:i + nBatch] for i in range(0, len(img_list), nBatch)]
    
    with torch.no_grad():
        for idx, batch in enumerate(batch_list):            
            batch = [os.path.join(args.input[0], i) for i in batch]

            inputs = []
            for b in batch:
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
                
                inputs.append(input)
            
            result = model(inputs)

            if cfg.TEST.AUG.ENABLED:
                result = [result]
       
            for i in range(len(result[0])):
                filename = os.path.basename(os.path.splitext(batch[i])[0])

                outxmlfilepath = os.path.join(args.output, 'out_xml')
                outxmlfilepath = os.path.join(outxmlfilepath, filename + '.xml')
                write_xml([inputs[i]["height"], inputs[i]["width"], 3], result[0][i], metadata.thing_classes, outxmlfilepath)


                if args.save_outimg:
                    outimgfilepath = os.path.join(args.output, 'out_img')
                    outimgfilepath = os.path.join(outimgfilepath, filename + '.jpg')

                    image = inputs[i]["image"].permute(1, 2, 0).numpy().astype("uint8")[:, :, ::-1]                
                    visualizer = Visualizer(image, metadata, instance_mode=ColorMode.IMAGE)
                    if "instances" in result[0][i]:
                        instances = result[0][i]["instances"].to(torch.device("cpu"))
                        vis_output = visualizer.draw_instance_predictions(predictions=instances)
                        vis_output.save(outimgfilepath)
            
            print('[{}/{}]'.format((idx+1)*len(batch), nImgs))  

    print('All inferences have been finished')
    