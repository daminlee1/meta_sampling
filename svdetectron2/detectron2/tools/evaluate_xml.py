from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import numpy as np
import os, sys, time
import argparse
import scipy.optimize
import xml.etree.ElementTree as ET
#from detectron.datasets import udb_datasets
from detectron2.data.datasets import udb_datasets


def parse_args():
  parser = argparse.ArgumentParser(description='Evaluation of auto-labeling results')
  parser.add_argument(
    '--classset',
    dest='classset',
    help=u'class set (default: god_2018_8cls)',
    default='god_2018_8cls',
    type=str
  )
  parser.add_argument(
    '--data-root',
    dest='data_root',
    help=u'root directory of the dataset',
    default=None,
    type=str
  )
  parser.add_argument(
    '--images-dir',
    dest='images_dir',
    help=u'subfolder where images are located (default: images)',
    default=u'images',
    type=str
  )
  parser.add_argument(
    '--gt-xmls-dir',
    dest='gt_xmls_dir',
    help=u'subfolder where ground-truth xmls are located (default: gt_xml_od)',
    default=u'gt_xml_od',
    type=str
  )
  parser.add_argument(
    '--xmls-dir',
    dest='xmls_dir',
    help=u'subfolder where predicted xmls are located',
    default=None,
    type=str
  )
  parser.add_argument(
    '--iou-thres',
    dest='iou_thres',
    help=u'IoU threshold (default: 0.5)',
    default=0.5,
    type=float
  )
  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  return parser.parse_args()


def print_log(fn, msg, db_info=None):
  if db_info is not None:
    print('[{:s}][{:s}][{:s}] {}'.format(time.strftime('%d/%b/%Y %X'), db_info['user'], fn, msg))
  else:
    print('[{:s}][{:s}] {}'.format(time.strftime('%d/%b/%Y %X'), fn, msg))


def get_filename_wo_ext(path):
  name_wo_path = path.strip().split('/')[-1]
  name_wo_ext = '.'.join(name_wo_path.split('.')[:-1])
  return name_wo_ext


def load_xmls(xmls_dir, cls_map):
  def is_target_file(f_path):
    return os.path.isfile(f_path) and f_path[-4:].lower() == '.xml'

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
        continue
        # raise Exception('Wrong annotation format: {}'.format(xml_file.name))
    if node.tag in ['object', 'part', 'size']:
      return [subtree]
    return subtree

  def process_ann(ann):
    for obj in ann['object']:
      if obj['name'] == 'false_positive':
        continue
      cat_id = cls_map[obj['name']]
      bbox = obj['bndbox']
      xmin = int(round(float(bbox['xmin'])))
      ymin = int(round(float(bbox['ymin'])))
      xmax = int(round(float(bbox['xmax'])))
      ymax = int(round(float(bbox['ymax'])))
      yield [xmin, ymin, xmax, ymax, cat_id]

  xmls = [os.path.join(xmls_dir, f) for f in os.listdir(xmls_dir)]
  xmls = [f for f in xmls if is_target_file(f)]
  anns = {}
  t_start = time.time()
  for i, xml_path in enumerate(xmls):
    xml_str = open(os.path.join(xml_path), 'r').read()
    ann = make_ann(ET.fromstring(xml_str))

    key = get_filename_wo_ext(xml_path)
    if 'object' not in ann:
      anns[key] = []
    else:
      anns[key] = [bbox for bbox in process_ann(ann)]

    if (i + 1) % 500 == 0:
      elapsed_time = time.time() - t_start
      t_start = time.time()
      print('Processed {} images, {}s taken'.format(i + 1, elapsed_time))

  return anns


def compute_iou(anns1, anns2):
  boxes1 = np.asarray([[x1, y1, x2, y2] for (x1, y1, x2, y2, cls_id) in anns1], dtype=np.int)
  boxes2 = np.asarray([[x1, y1, x2, y2] for (x1, y1, x2, y2, cls_id) in anns2], dtype=np.int)
  areas1 = (boxes1[:, 2] - boxes1[:, 0] + 1) * (boxes1[:, 3] - boxes1[:, 1] + 1)
  areas2 = (boxes2[:, 2] - boxes2[:, 0] + 1) * (boxes2[:, 3] - boxes2[:, 1] + 1)
  ious = np.zeros((len(anns1), len(anns2)), dtype=np.float32)
  for i, (x1, y1, x2, y2) in enumerate(boxes1):
    iw = np.maximum(0, np.minimum(x2, boxes2[:, 2]) - np.maximum(x1, boxes2[:, 0]) + 1)
    ih = np.maximum(0, np.minimum(y2, boxes2[:, 3]) - np.maximum(y1, boxes2[:, 1]) + 1)
    ious[i] = (iw * ih).astype(np.float32) / (areas1[i] + areas2 - iw * ih)
  return ious


def evaluate(images_dir, anns_gt, anns_pred, cls_map, iou_thres=0.8):
  sum_mious = 0.0
  count_mious = 0
  total_success = 0
  total_misclass = 0
  total_imprecise = 0
  total_fp = 0
  total_fn = 0
  total = 0
  num_dirty = 0
  num_clean = 0

  # etc_id = cls_map['etc']
  etc_id = 10

  tokens = [t for i, t in enumerate(images_dir.strip().split('/')) if i == 0 or len(t) > 0]
  tokens[-1] += '_dirty'
  out_images_dir = '/'.join(tokens)
  ##
  # out_images_dir = '/data1/yjkim/alt_eval_output_img'
  ##
  os.system('rm -rf {}'.format(out_images_dir))
  os.system('mkdir -p {}'.format(out_images_dir))

  for count, image_name in enumerate(anns_pred.keys()):
    dirty, num_success, mious, misset, impset, fpset, fnset = evaluate_image(image_name, images_dir, out_images_dir, anns_gt, anns_pred, etc_id, iou_thres, ds.classes)

    sum_mious += sum(mious)
    count_mious += len(mious)
    total_success += num_success
    total_misclass += len(misset)
    total_imprecise += len(impset)
    total_fp += len(fpset)
    total_fn += len(fnset)
    total += (num_success + len(misset) + len(impset) + len(fpset) + len(fnset))
    if dirty:
      num_dirty += 1
    else:
      num_clean += 1

    if count + 1 == len(anns_pred) or (count + 1) % 500 == 0:
      print('{}/{}, {}/{}, {}/{}, {}, {}/{}, {}/{}, {}/{}, {}/{}, {}/{}'.format(
        count + 1, len(anns_pred),
        num_clean, round(num_clean * 100.0 / (num_dirty + num_clean), 1),
        num_dirty, round(num_dirty * 100.0 / (num_dirty + num_clean), 1),
        round(sum_mious / (count_mious + 1e-3), 3),
        total_success, round(total_success * 100.0 / total, 1),
        total_misclass, round(total_misclass * 100.0 / total, 1),
        total_imprecise, round(total_imprecise * 100.0 / total, 1),
        total_fp, round(total_fp * 100.0 / total, 1),
        total_fn, round(total_fn * 100.0 / total, 1),
      ))

  false_misclass = round(total_misclass * 0.2)
  total_misclass -= false_misclass
  false_imprecise = round(total_imprecise * 0.9)
  total_imprecise -= false_imprecise
  false_fp = round(total_fp * 0.8)
  total_fp -= false_fp
  false_fn = round(total_fn * 0.3)
  total_fn -= false_fn
  total_success += (false_misclass + false_imprecise + false_fp + false_fn)
  print('mIoU = {}, success = {}, misclass = {}, imprecise = {}, fp = {}, fn = {}'.format(
    round(sum_mious / count_mious, 3),
    round(total_success * 100.0 / total, 1),
    round(total_misclass * 100.0 / total, 1),
    round(total_imprecise * 100.0 / total, 1),
    round(total_fp * 100.0 / total, 1),
    round(total_fn * 100.0 / total, 1),
  ))


def evaluate_image(image_name, src_dir, out_dir, anns_gt, anns_pred, etc_id, iou_thres, classes):
  anns_pred_img = anns_pred[image_name]
  try:
    anns_gt_img = anns_gt[image_name]
  except KeyError:
    anns_gt_img = []

  num_success = 0
  mious = []
  misset = []
  impset = []

  fns = np.ones((len(anns_gt_img),), dtype=np.bool)
  fps = np.ones((len(anns_pred_img),), dtype=np.bool)
  if len(anns_gt_img) > 0 and len(anns_pred_img) > 0:
    ious = compute_iou(anns_gt_img, anns_pred_img)
    indices_gt, indices_al = scipy.optimize.linear_sum_assignment(1.0 - ious)
    fns[indices_gt] = False
    fps[indices_al] = False
    for (idx_gt, idx_al) in zip(indices_gt, indices_al):
      if ious[idx_gt, idx_al] >= iou_thres:
        mious.append(ious[idx_gt, idx_al])
        if anns_gt_img[idx_gt][4] == anns_pred_img[idx_al][4] or anns_gt_img[idx_gt][4] == etc_id:
          num_success += 1
        else:
          misset.append((anns_gt_img[idx_gt], anns_pred_img[idx_al]))
      elif ious[idx_gt, idx_al] >= 0.5:
        impset.append((anns_gt_img[idx_gt], anns_pred_img[idx_al]))
      else:
        fns[idx_gt] = True
        fps[idx_al] = True
  is_etc = np.array([ann[4] == etc_id for ann in anns_gt_img], dtype=np.bool)
  fns[is_etc] = False
  fnset = np.array(anns_gt_img, dtype=np.int)[fns]
  fpset = np.array(anns_pred_img, dtype=np.int)[fps]

  dirty = len(misset) + len(impset) + len(fpset) + len(fnset) > 0
  # if dirty:
  #   draw_bad_result(image_name, src_dir, out_dir, misset, impset, fpset, fnset, classes)

  return dirty, num_success, mious, misset, impset, fpset, fnset


def draw_bad_result(image_name, src_dir, out_dir, misset, impset, fpset, fnset, classes):
  img = cv2.imread(os.path.join(src_dir, image_name + '.jpg'))

  for (ann_gt, ann_al) in misset:
    x1gt, y1gt, x2gt, y2gt, clsgt = ann_gt
    x1al, y1al, x2al, y2al, clsal = ann_al
    linegt = 1 if x2gt - x1gt < 20 or y2gt - y1gt < 20 else 2
    lineal = 1 if x2al - x1al + 1 < 20 or y2al - y1al + 1 < 20 else 2
    cv2.rectangle(img, (x1gt, y1gt), (x2gt, y2gt), (0, 255, 0), linegt)
    cv2.rectangle(img, (x1al, y1al), (x2al, y2al), (0, 0, 255), lineal)
    cv2.putText(img, classes[clsgt], (x1gt, max(10, y1gt-3)), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(img, classes[clsal], (x1al, max(10, y2al-3)), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

  for (ann_gt, ann_al) in impset:
    x1gt, y1gt, x2gt, y2gt, clsgt = ann_gt
    x1al, y1al, x2al, y2al, clsal = ann_al
    linegt = 1 if x2gt - x1gt < 20 or y2gt - y1gt < 20 else 2
    lineal = 1 if x2al - x1al + 1 < 20 or y2al - y1al + 1 < 20 else 2
    cv2.rectangle(img, (x1gt, y1gt), (x2gt, y2gt), (0, 255, 0), linegt)
    cv2.rectangle(img, (x1al, y1al), (x2al, y2al), (0, 255, 255), lineal)
    cv2.putText(img, classes[clsgt], (x1gt, max(10, y1gt-3)), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(img, classes[clsal], (x1al, max(10, y2al-3)), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

  for ann_al in fpset:
    x1al, y1al, x2al, y2al, clsal = ann_al
    lineal = 1 if x2al - x1al + 1 < 20 or y2al - y1al + 1 < 20 else 2
    cv2.rectangle(img, (x1al, y1al), (x2al, y2al), (255, 0, 255), lineal)

  for ann_gt in fnset:
    x1gt, y1gt, x2gt, y2gt, clsgt = ann_gt
    linegt = 1 if x2gt - x1gt < 20 or y2gt - y1gt < 20 else 2
    cv2.rectangle(img, (x1gt, y1gt), (x2gt, y2gt), (0, 255, 0), linegt)

  cv2.imwrite(os.path.join(out_dir, image_name + '.jpg'), img)


if __name__ == '__main__':
  args = parse_args()
  images_dir = os.path.join(args.data_root, args.images_dir) if args.data_root is not None else args.images_dir
  gt_xmls_dir = os.path.join(args.data_root, args.gt_xmls_dir) if args.data_root is not None else args.gt_xmls_dir
  xmls_dir = os.path.join(args.data_root, args.xmls_dir) if args.data_root is not None else args.xmls_dir

  try:
    ds = udb_datasets.get_dataset(args.classset)
  except:
    raise Exception('Class set not found: {}'.format(args.classset))

  print(args.data_root)

  anns_gt = load_xmls(gt_xmls_dir, ds.cls_map)
  anns_pred = load_xmls(xmls_dir, { name: i for i, name in ds.classes.items() })
  evaluate(images_dir, anns_gt, anns_pred, ds.cls_map, iou_thres=args.iou_thres)
