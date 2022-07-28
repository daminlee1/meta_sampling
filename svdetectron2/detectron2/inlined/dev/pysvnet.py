from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import numpy as np
import os, sys, time
import random, string
import json
import ctypes

#from detectron.app.common import print_log, connect_to_server, recv_large, send_large, prepare_file, report_error, report_finish, get_video_frames
from app_common import print_log, connect_to_server, recv_large, send_large, prepare_file, report_error, report_finish, get_video_frames

if sys.version_info[0] < 3:
  from Queue import Queue as queue
else:
  from queue import Queue as queue
  from builtins import str

if cv2.__version__.startswith('2.'):
  CV_LOAD_IMAGE_COLOR = cv2.CV_LOAD_IMAGE_COLOR
else:
  CV_LOAD_IMAGE_COLOR = cv2.IMREAD_COLOR


class CommandLineParams(ctypes.Structure):
  _fields_ = [
    ('det', ctypes.c_bool),
    ('det_3d', ctypes.c_bool),
    ('det_tstl', ctypes.c_bool),
    ('class_tsr', ctypes.c_bool),
    ('seg_fsd', ctypes.c_bool),
    ('seg_lane', ctypes.c_bool),
    ('tiling', ctypes.c_bool),
    ('temporal_filter', ctypes.c_bool),
    ('sampling_out', ctypes.c_bool),
    ('skip', ctypes.c_int),
    ('test_frames', ctypes.c_int),
    ('time_check', ctypes.c_bool),
    ('preview', ctypes.c_int),
    ('imshow', ctypes.c_bool),
    ('yuyv', ctypes.c_bool),
  ]


class SVNetPythonOutputLane(ctypes.Structure):
  MAX_DEGREE_VEHICLE_POLY = 11
  MAX_DEGREE_IMAGE_POLY = 11

  _fields_ = [
    ('poly_yw_degree', ctypes.c_int),
    ('poly_yw_param', ctypes.c_double * (MAX_DEGREE_VEHICLE_POLY + 1)),
    ('imageTopy', ctypes.c_int),
    ('imageBottomy', ctypes.c_int),
    ('poly_yx_degree', ctypes.c_int),
    ('poly_yx_param', ctypes.c_double * (MAX_DEGREE_IMAGE_POLY + 1)),
    ('poly_yx_mean', ctypes.c_double),
    ('poly_yx_inv_sigma', ctypes.c_double),
    ('type1', ctypes.c_int),
    ('type2', ctypes.c_int),
    ('type3', ctypes.c_int),
    ('type4', ctypes.c_int),
  ]

  def to_dict(self):
    dct = {
      'ymin': self.imageTopy,
      'ymax': self.imageBottomy,
      'ymean': self.poly_yx_mean,
      'yscale': self.poly_yx_inv_sigma,
      'wparam': self.poly_yw_param[:self.poly_yw_degree+1],
      'xparam': self.poly_yx_param[:self.poly_yx_degree+1],
    }
    return dct


class SVNetPythonOutput(ctypes.Structure):
  _fields_ = [
    ('fs_rle', ctypes.POINTER(ctypes.c_int)),
    ('fs_rle_len', ctypes.c_int),
    ('fs', ctypes.POINTER(ctypes.c_ubyte)),
    ('lanes', ctypes.POINTER(SVNetPythonOutputLane)),
    ('num_lanes', ctypes.c_int),
  ]


def initialize(model_path, task_type):
  cfgs = {
    'ld': { 'net': 'svnetl3206l', 'ini': 'options.ini', 'params_fn': get_params_LD },
    'fsd': { 'net': 'svnetl400f', 'ini': 'options.ini', 'params_fn': get_params_FSD },
  }
  cfg = cfgs[task_type]
  svnet_lib = ctypes.cdll.LoadLibrary(model_path)
  net = svnet_lib.create_instance(ctypes.c_char_p(cfg['net']), ctypes.c_char_p('gpu'), 0, ctypes.c_char_p(cfg['ini']))
  params = cfg['params_fn']()
  return svnet_lib, net, params


def delete(svnet_lib, net):
  svnet_lib.delete_instance(net)


def get_params_LD():
  params = CommandLineParams()
  params.det = 0
  params.det_3d = 0
  params.det_tstl = 0
  params.class_tsr = 0
  params.seg_fsd = 0
  params.seg_lane = 1
  params.tiling = 0
  params.temporal_filter = 0
  params.sampling_out = 0
  params.skip = 0
  params.test_frames = -1
  params.time_check = 0
  params.preview = 0
  params.imshow = 0
  params.yuyv = 0
  return params


def get_params_FSD():
  params = CommandLineParams()
  params.det = 0
  params.det_3d = 0
  params.det_tstl = 0
  params.class_tsr = 0
  params.seg_fsd = 1
  params.seg_lane = 0
  params.tiling = 0
  params.temporal_filter = 0
  params.sampling_out = 0
  params.skip = 0
  params.test_frames = -1
  params.time_check = 0
  params.preview = 0
  params.imshow = 0
  params.yuyv = 0
  return params


def main(cfg_path, task_type):
  #try:
    cfg = json.loads(open(cfg_path, 'r').read())
    conn = connect_to_server(cfg['server']['host'], cfg['server']['port'])
    svnet_lib, net, params = initialize('./svnet.so', task_type)
    process(svnet_lib, net, params, conn, cfg['media_root'], task_type)
    delete(svnet_lib, net)
    conn.close()

  #except KeyboardInterrupt as e:
  #  pass
  #except Exception as e:
  #  pass


def execute_svnet(svnet_lib, net, params, img, tempkey):
  out = None
  anns = []

  try:
    out = SVNetPythonOutput()
    cv2.imwrite('./__temp_{}__.png'.format(tempkey), img)
    imgbin = open('./__temp_{}__.png'.format(tempkey), 'rb').read()
    svnet_lib.test_input2(net, imgbin, len(imgbin), None, ctypes.byref(params), ctypes.byref(out))

    if out.fs_rle_len > 0:
      height, width = img.shape[:2]
      #json_str = json.dumps({ 'mask': out.fs_rle[:out.fs_rle_len] })
      mask = np.array(out.fs[:height*width], dtype=np.uint8).reshape((height, width))
      mask_str = contour_to_text(mask_to_contour(mask, 4))
      json_str = json.dumps({ 'mask': mask_str })
      anns = [json_str]

    elif out.num_lanes > 0:
      anns = []
      for lane in out.lanes[:out.num_lanes]:
        lane_dict = lane.to_dict()
        lane_dict['mask'] = points_to_text(lane_to_points(lane_dict))
        anns.append(json.dumps(lane_dict))

  finally:
    if out is not None:
      svnet_lib.delete_python_output(ctypes.byref(out))
    return anns


def process_task_video(svnet_lib, net, params, conn, task_data, file_path, task_type, tempkey):
  file_type = task_data['file_type']
  file_id = task_data['video_id']
  start_frame_id = task_data['start_frame_id']
  end_frame_id = task_data['end_frame_id']
  label_interval = task_data['label_interval']

  for (frame_id, img) in get_video_frames(file_path, start_frame_id, end_frame_id, label_interval):
    anns = execute_svnet(svnet_lib, net, params, img, tempkey)
    req_data = {
      'req_type': 'post_ann',
      'task_type': task_type,
      'file_type': file_type,
      'video_id': file_id,
      'frame_id': frame_id,
      'anns': anns,
    }
    send_large(conn, req_data)
    _ = recv_large(conn)


def process_task_image(svnet_lib, net, params, conn, task_data, file_path, task_type, tempkey):
  file_type = task_data['file_type']
  file_id = task_data['image_id']
  width = task_data['width']
  height = task_data['height']

  img = cv2.imread(file_path, CV_LOAD_IMAGE_COLOR)
  anns = execute_svnet(svnet_lib, net, params, img, tempkey)
  req_data = {
    'req_type': 'post_ann',
    'task_type': task_type,
    'file_type': file_type,
    'image_id': file_id,
    'anns': anns,
  }
  send_large(conn, req_data)
  _ = recv_large(conn)


def process(svnet_lib, net, params, conn, media_root, task_type):
  req_data = { 'req_type': 'get_task', 'task_type': task_type }
  recv_data = {}
  tempkey = ''.join(random.choice(string.ascii_letters) for x in range(10))

  while 'terminate' not in recv_data:
    try:
      send_large(conn, req_data)
      recv_data = recv_large(conn)

      if recv_data is None:
        break

      if 'task' not in recv_data:
        time.sleep(1)
        continue

      task_data = recv_data['task']
      file_path = prepare_file(conn, media_root, task_data)
      if file_path is None:
        report_error(conn, task_data, 'file_not_found')
        continue

      if task_data['file_type'] == 'video':
        process_task_video(svnet_lib, net, params, conn, task_data, file_path, task_type, tempkey)
      else:
        process_task_image(svnet_lib, net, params, conn, task_data, file_path, task_type, tempkey)

      report_finish(conn, task_data)

    except Exception as e:
      print_log('process',
          'Error occurred from recv_data [{}]: {}'.format(recv_data, repr(e)))
      time.sleep(5)
      continue


def mask_to_contour(mask, step):
  if cv2.__version__[0] in ['2', '4']:
    contours, hiers = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  else:
    _, contours, hiers = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  contours = [contour.astype(int) for contour in contours if contour.shape[0] >= 3]
  for idx in range(len(contours)):
    #epsilon = 0.001 * cv2.arcLength(contours[idx], True)
    #contours[idx] = cv2.approxPolyDP(contours[idx], epsilon, True)
    if step > 1 and contours[idx].shape[0] > step*4:
      contours[idx] = contours[idx][::step*2]
    elif step > 1 and contours[idx].shape[0] > step*2:
      contours[idx] = contours[idx][::step]
  return contours


def contour_to_text(contour):
  text = '|'.join([','.join([str(val) for val in cnt_child.reshape(-1)])
                  for cnt_child in contour])
  return text


def points_to_text(points):
  text = ','.join(['{},{}'.format(x, y) for (x, y) in points])
  return text


def contour_from_text(contour_text):
  contour = []
  for cnt_child_text in contour_text.split('|'):
    cnt_child = [int(val) for val in cnt_child_text.split(',')]
    cnt_child = np.array(cnt_child, dtype=np.int32).reshape((-1, 1, 2))
    contour.append(cnt_child)
  return contour


def lane_to_points(lane):
  ymin = int(lane['ymin'])
  ymax = int(lane['ymax'])
  ymean = float(lane['ymean'])
  yscale = float(lane['yscale'])
  wparam = [float(val) for val in lane['wparam']]
  xparam = [float(val) for val in lane['xparam']]

  lpoints = []
  rpoints = []
  for y in range(ymin, ymax+ 1, 4):
    y_ = (y - ymean) * yscale
    pow_y = 1.0
    x = 0.0
    for p in xparam:
      x += p * pow_y
      pow_y *= y_
    w = y * wparam[1] + wparam[0]

    lpoints.append((int(round(x - w / 2.0)), int(round(y))))
    rpoints.append((int(round(x + w / 2.0)), int(round(y))))
  return lpoints + rpoints[::-1]


if __name__ == '__main__':
  main('config.json', sys.argv[1])
