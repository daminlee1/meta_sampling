from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import os, sys, time
import xml.etree.ElementTree as ET
import json

import detectron2.data.datasets.udb_datasets as udb_datasets

if sys.version_info[0] < 3:
  from Queue import Queue as queue
else:
  from queue import Queue as queue
  from builtins import str


def parse_args():
  parser = argparse.ArgumentParser(description='Converter from XML annotations (VOC format) to COCO json format')
  parser.add_argument(
    '--dataset',
    dest='dataset',
    help='dataset (coco, cityscapes, god, god_2018, god_2018_4cls)',
    default='god_2018',
    type=str
  )
  parser.add_argument(
    '--root-dir',
    dest='root_dir',
    help='root directory where datasets are located (for processing multiple datasets only)',
    default=None,
    type=str
  )
  parser.add_argument(
    '--images-dir',
    dest='images_dir',
    help='subfolder where image files are located',
    default=None,
    type=str
  )
  parser.add_argument(
    '--xmls-dir',
    dest='xmls_dir',
    help='subfolder where xml files are located',
    default=None,
    type=str
  )
  parser.add_argument(
    '--output-path',
    dest='output_path',
    help='path to json output file',
    default=None,
    type=str
  )
  parser.add_argument(
    '--occ-thres',
    dest='occ_thres',
    help='occlusion threshold',
    default=None,
    type=float
  )
  parser.add_argument(
    '--trunc-thres',
    dest='trunc_thres',
    help='truncation threshold',
    default=None,
    type=float
  )
  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  return parser.parse_args()


def get_categories(classes):
  def if_etc_to_ignore(cls_name):
    return 'ignored' if cls_name == 'etc' else cls_name

  cls_indices = list(classes.keys())
  cls_indices.sort()
  out = [
    { u'id': cls_idx, u'name': u'{}'.format(if_etc_to_ignore(classes[cls_idx])) }
    for cls_idx in cls_indices if cls_idx > 0
  ]
  return out


def get_xml_img_pairs(images_dir, xmls_dir):
  def get_filename_wo_ext(path):
    name_wo_path = path.strip().split('/')[-1]
    name_wo_ext = '.'.join(name_wo_path.split('.')[:-1])
    return name_wo_ext

  img_names = {
    get_filename_wo_ext(f): f
    for f in os.listdir(images_dir) if f[-4:].lower() in ['.jpg', '.png']
  }
  xml_names = [f for f in os.listdir(xmls_dir) if f[-4:].lower() == '.xml']

  out = []
  for xml_name in xml_names:
    try:
      img_name = img_names[get_filename_wo_ext(xml_name)]
      out.append((xml_name, img_name))
    except:
      print('Image file does not exist: {}'.format(xml_name))

  return out


def convert(images_dir, xmls_dir, dataset, output_path, occ_thres, trunc_thres):
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

  def process_ann(ann):
    for obj in ann['object']:
      if obj['name'].startswith('false_positive'):
        continue
      cat_id = ds.cls_map[obj['name']]
      if occ_thres is not None and trunc_thres is not None:
        if 'occluded' not in obj or float(obj['occluded']) > occ_thres or \
           'truncated' not in obj or float(obj['truncated']) > trunc_thres:
          cat_id = ignored_id
      bbox = obj['bndbox']
      xmin = round(float(bbox['xmin']), 1)
      ymin = round(float(bbox['ymin']), 1)
      xmax = round(float(bbox['xmax']), 1)
      ymax = round(float(bbox['ymax']), 1)
      yield xmin, ymin, xmax, ymax, cat_id

  ds = udb_datasets.get_dataset(dataset)
  categories = get_categories(ds.classes)
  ignored_id = ds.cls_map['ignored']

  pairs = get_xml_img_pairs(images_dir, xmls_dir)
  images = []
  empty_images = []
  annotations = []
  t_start = time.time()
  for img_id, (xml_name, img_name) in enumerate(pairs):
    xml_str = open(os.path.join(xmls_dir, xml_name), 'r').read()
    ann = make_ann(ET.fromstring(xml_str))

    if not 'size' in ann or len(ann['size']) != 1:
      raise Exception('Wrong annotation format: {}'.format(xml_name))

    im_width = int(ann['size'][0]['width'])
    im_height = int(ann['size'][0]['height'])
    img_dict = {
      'file_name': '{}'.format(img_name),
      'id': img_id,
      'width': im_width,
      'height': im_height,
    }

    # if not 'object' in ann:
    #   empty_images.append(img_dict)
    #   continue
    # else:
    #   images.append(img_dict)

    images.append(img_dict)
    if not 'object' in ann:
      ann['object'] = []

    for xmin, ymin, xmax, ymax, cat_id in process_ann(ann):
      width = max(0, round(min(im_width - 1, xmax) - xmin + 1.0, 1))
      height = max(0, round(min(im_height - 1, ymax) - ymin + 1.0, 1))
      area = round(width * height, 2)
      obj_dict = {
        'segmentation': [],
        'area': area,
        'iscrowd': 0,
        'image_id': img_id,
        'bbox': [xmin, ymin, width, height],
        'category_id': cat_id,
        'id': len(annotations),
      }
      annotations.append(obj_dict)

    if (img_id + 1) % 100 == 0:
      elapsed_time = time.time() - t_start
      t_start = time.time()
      print('Processed {} images, {}s taken'.format(img_id + 1, elapsed_time))

  out = {
    'images': images,
    'empty_images': empty_images,
    'annotations': annotations,
    'categories': categories,
  }

  with open(output_path, 'w') as f:
    f.write(json.dumps(out))
  return out


if __name__ == '__main__':
  args = parse_args()
  if args.root_dir is not None:
    db_names = [f for f in os.listdir(args.root_dir) if os.path.isdir(os.path.join(args.root_dir, f))]
    for db_name in db_names:
      images_dir = os.path.join(args.root_dir, db_name, args.images_dir)
      xmls_dir = os.path.join(args.root_dir, db_name, args.xmls_dir)
      if not os.path.isdir(images_dir) or not os.path.isdir(xmls_dir):
        print('Images and/or XMLs subfolders not found: {}'.format(db_name))
        continue
      output_path = os.path.join(args.root_dir, 'annotations_{}.json'.format(db_name))
      print('Processing {}...'.format(db_name))
      convert(images_dir, xmls_dir, args.dataset, output_path)
      print('Finished {}'.format(db_name))
  else:
    convert(args.images_dir, args.xmls_dir, args.dataset, args.output_path, args.occ_thres, args.trunc_thres)
