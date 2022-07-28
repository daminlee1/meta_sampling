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


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)


    # args.input = glob.glob(os.path.expanduser(args.input[0]))

    with open(os.path.join(args.input[0], 'ImageSets/all.txt')) as f:
        img_list = [line.rstrip() + '.jpg' for line in f]


    for i in range(len(img_list)):
        path = os.path.join(args.input[0], 'JPEGImages')
        path = os.path.join(path, img_list[i])
        img = read_image(path, format="BGR")
        predictions, visualized_output = demo.run_on_image(img)

        im2 = cv2.imread(path)

        print('inference on ' + str(i) + ' image')

        for j in range(len(predictions[0][0]['instances'])):
            bbox = predictions[0][0]['instances'][j].pred_boxes.tensor.cpu().numpy()
            xmin = bbox[0][0]
            ymin = bbox[0][1]
            xmax = bbox[0][2]
            ymax = bbox[0][3]
            score = predictions[0][0]['instances'][j].scores.cpu().numpy()[0]
            dcls = predictions[0][0]['instances'][j].pred_classes.tolist()[0] + 1

            # cv2.rectangle(im2, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)


        boxes           = predictions[1].pred_boxes.tensor.cpu().tolist()
        boxes_3d        = predictions[1].boxes_3d.cpu().tolist()
        dir_scores      = predictions[1].dir_scores.cpu().tolist()
        pred_directions = predictions[1].pred_directions.cpu().tolist()

        for j in range(len(predictions[1])):
            x1 = boxes[j][0]
            y1 = boxes[j][1]
            x2 = boxes[j][2]
            y2 = boxes[j][3]

            w = x2 - x1 + 1
            h = y2 - y1 + 1

            # num points == 16
            x = [x1, x2, x1, x2, x1, x2, x1, x2, x1, x2, x1, x2, x1, x2, x1, x2]
            y = [y1, y1, y2, y2, y1, y1, y2, y2, y1, y1, y2, y2, y1, y1, y2, y2]
            p = [0] * 32
            xn = [0] * 8
            yn = [0] * 8

            for k in range(16):
                if x[k] == -1:
                    break
                else:
                    im_size = predictions[0][0]['instances'].image_size
                    p[k*2+0] = max(min(w+x[k], im_size[1] - 1), 0)
                    p[k*2+1] = max(min(w+y[k], im_size[0] - 1), 0)

                    if pred_directions[j] == 1 or pred_directions[j] == 5:
                        xn[0] = p[0]
                        yn[0] = p[1]
                        xn[1] = p[2]
                        yn[1] = p[3]
                        xn[2] = p[4]
                        yn[2] = p[5]
                        xn[3] = p[6]
                        yn[3] = p[7]
                        xn[4] = p[8]
                        yn[4] = p[9]
                        xn[5] = p[10]
                        yn[5] = p[11]
                        xn[6] = p[12]
                        yn[6] = p[13]
                        xn[7] = p[14]
                        yn[7] = p[15]

                    elif pred_directions[j] == 3 or pred_directions[j] == 7:
                        if p[21] + p[29] >= p[23] + p[31]:
                            xn[0] = (p[16] + p[24]) / 2
                            yn[0] = (p[17] + p[25]) / 2
                            xn[1] = (p[16] + p[24]) / 2
                            yn[1] = (p[17] + p[25]) / 2
                            xn[2] = (p[20] + p[28]) / 2
                            yn[2] = (p[21] + p[29]) / 2
                            xn[3] = (p[20] + p[28]) / 2
                            yn[3] = (p[21] + p[29]) / 2
                            xn[4] = (p[18] + p[26]) / 2
                            yn[4] = (p[19] + p[27]) / 2
                            xn[5] = (p[18] + p[26]) / 2
                            yn[5] = (p[19] + p[27]) / 2
                            xn[6] = (p[22] + p[30]) / 2
                            yn[6] = (p[23] + p[31]) / 2
                            xn[7] = (p[22] + p[30]) / 2
                            yn[7] = (p[23] + p[31]) / 2
                        else:
                            xn[0] = (p[18] + p[26]) / 2
                            yn[0] = (p[19] + p[27]) / 2
                            xn[1] = (p[18] + p[26]) / 2
                            yn[1] = (p[19] + p[27]) / 2
                            xn[2] = (p[22] + p[30]) / 2
                            yn[2] = (p[23] + p[31]) / 2
                            xn[3] = (p[22] + p[30]) / 2
                            yn[3] = (p[23] + p[31]) / 2
                            xn[4] = (p[16] + p[24]) / 2
                            yn[4] = (p[17] + p[25]) / 2
                            xn[5] = (p[16] + p[24]) / 2
                            yn[5] = (p[17] + p[25]) / 2
                            xn[6] = (p[20] + p[28]) / 2
                            yn[6] = (p[21] + p[29]) / 2
                            xn[7] = (p[20] + p[28]) / 2
                            yn[7] = (p[21] + p[29]) / 2
                    elif pred_directions[j] == 2 or pred_directions[j] == 6:
                        xn[0] = (p[0] + p[24]) / 2
                        yn[0] = (p[1] + p[25]) / 2
                        xn[1] = (p[2] + p[16]) / 2
                        yn[1] = (p[3] + p[17]) / 2
                        xn[2] = (p[4] + p[28]) / 2
                        yn[2] = (p[5] + p[29]) / 2
                        xn[3] = (p[6] + p[20]) / 2
                        yn[3] = (p[7] + p[21]) / 2
                        xn[4] = (p[8] + p[26]) / 2
                        yn[4] = (p[9] + p[27]) / 2
                        xn[5] = (p[10] + p[18]) / 2
                        yn[5] = (p[11] + p[19]) / 2
                        xn[6] = (p[12] + p[30]) / 2
                        yn[6] = (p[13] + p[31]) / 2
                        xn[7] = (p[14] + p[22]) / 2
                        yn[7] = (p[15] + p[23]) / 2
                    else:
                        xn[0] = (p[0] + p[26]) / 2
                        yn[0] = (p[1] + p[27]) / 2
                        xn[1] = (p[2] + p[18]) / 2
                        yn[1] = (p[3] + p[19]) / 2
                        xn[2] = (p[4] + p[30]) / 2
                        yn[2] = (p[5] + p[31]) / 2
                        xn[3] = (p[6] + p[22]) / 2
                        yn[3] = (p[7] + p[23]) / 2
                        xn[4] = (p[8] + p[24]) / 2
                        yn[4] = (p[9] + p[25]) / 2
                        xn[5] = (p[10] + p[16]) / 2
                        yn[5] = (p[11] + p[17]) / 2
                        xn[6] = (p[12] + p[28]) / 2
                        yn[6] = (p[13] + p[29]) / 2
                        xn[7] = (p[14] + p[20]) / 2
                        yn[7] = (p[15] + p[21]) / 2

            # cv2.circle(im2, (np.int(xn[0]), np.int(yn[0])), 3, (0, 255, 0), -1)
            # cv2.circle(im2, (np.int(xn[1]), np.int(yn[1])), 3, (0, 255, 0), -1)
            # cv2.circle(im2, (np.int(xn[2]), np.int(yn[2])), 3, (0, 255, 0), -1)
            # cv2.circle(im2, (np.int(xn[3]), np.int(yn[3])), 3, (0, 255, 0), -1)
            # cv2.circle(im2, (np.int(xn[4]), np.int(yn[4])), 3, (0, 255, 0), -1)
            # cv2.circle(im2, (np.int(xn[5]), np.int(yn[5])), 3, (0, 255, 0), -1)
            # cv2.circle(im2, (np.int(xn[6]), np.int(yn[6])), 3, (0, 255, 0), -1)
            # cv2.circle(im2, (np.int(xn[7]), np.int(yn[7])), 3, (0, 255, 0), -1)

            p0 = (np.int(xn[0]), np.int(yn[0]))
            p1 = (np.int(xn[1]), np.int(yn[1]))
            p2 = (np.int(xn[2]), np.int(yn[2]))
            p3 = (np.int(xn[3]), np.int(yn[3]))
            p4 = (np.int(xn[4]), np.int(yn[4]))
            p5 = (np.int(xn[5]), np.int(yn[5]))
            p6 = (np.int(xn[6]), np.int(yn[6]))
            p7 = (np.int(xn[7]), np.int(yn[7]))

            cv2.line(im2, p0, p1, (0, 255, 0), 2)
            cv2.line(im2, p0, p2, (0, 255, 0), 2)
            cv2.line(im2, p1, p3, (0, 255, 0), 2)
            cv2.line(im2, p2, p3, (0, 255, 0), 2)

            if (p1[0] < p5[0]):
                cv2.line(im2, p1, p5, (0, 255, 0), 2)
                cv2.line(im2, p3, p7, (0, 255, 0), 2)
                cv2.line(im2, p5, p7, (0, 255, 0), 2)

            elif p0[0] > p4[0]:
                cv2.line(im2, p0, p4, (0, 255, 0), 2)
                cv2.line(im2, p2, p6, (0, 255, 0), 2)
                cv2.line(im2, p4, p6, (0, 255, 0), 2)

            if p0[1] > p4[1] and p1[1] > p5[1]:
                cv2.line(im2, p0, p4, (0, 255, 0), 2)
                cv2.line(im2, p1, p5, (0, 255, 0), 2)
                cv2.line(im2, p4, p5, (0, 255, 0), 2)

            cv2.line(im2, p4, p5, (0, 255, 0), 1)
            cv2.line(im2, p4, p6, (0, 255, 0), 1)
            cv2.line(im2, p5, p7, (0, 255, 0), 1)
            cv2.line(im2, p6, p7, (0, 255, 0), 1)
            cv2.line(im2, p0, p4, (0, 255, 0), 1)
            cv2.line(im2, p1, p5, (0, 255, 0), 1)
            cv2.line(im2, p2, p6, (0, 255, 0), 1)
            cv2.line(im2, p3, p7, (0, 255, 0), 1)

            cv2.imwrite('result_test.jpg', im2)
            print('drawing')




        # cv2.imwrite('result_test.jpg', im2)


    print('inference was finished!')
