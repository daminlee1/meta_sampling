from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os, sys, time
import socket, json, struct, base64, shutil
import numpy as np
import cv2

if sys.version_info[0] < 3:
  pass
else:
  from builtins import str

if cv2.__version__.startswith('2.'):
  CV_CAP_PROP_POS_FRAMES = cv2.cv.CV_CAP_PROP_POS_FRAMES
  CV_CAP_PROP_POS_MSEC = cv2.cv.CV_CAP_PROP_POS_MSEC
  CV_CAP_PROP_FPS = cv2.cv.CV_CAP_PROP_FPS
else:
  CV_CAP_PROP_POS_FRAMES = cv2.CAP_PROP_POS_FRAMES
  CV_CAP_PROP_POS_MSEC = cv2.CAP_PROP_POS_MSEC
  CV_CAP_PROP_FPS = cv2.CAP_PROP_FPS


def get_file_path(media_root, user_hash, hashcode):
  return os.path.join(media_root, user_hash, hashcode)


def print_log(fn, msg):
  print('[{:s}][{:s}] {}'.format(time.strftime('%d/%b/%Y %X'), fn, msg))


def divide_frames(num_frames, chunks=None, chunk_size=None):
  if chunks is not None:
    chunk_size = num_frames // chunks

  start_ids = list(range(0, num_frames, chunk_size))
  end_ids = list(range(chunk_size - 1, num_frames, chunk_size))
  if len(end_ids) == 0 or end_ids[-1] < num_frames - 1:
    end_ids.append(num_frames - 1)

  if len(start_ids) > 1 and end_ids[-1] - start_ids[-1] + 1 < chunk_size / 2:
    end_ids[-2] = end_ids[-1]
    start_ids = start_ids[:-1]
    end_ids = end_ids[:-1]

  return start_ids, end_ids


def connect_to_server(host, port):
  conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  conn.connect((host, port))
  return conn


def recv_large(conn, chunk_size=4096):
  def recvall(conn, n):
    msg_queue = []
    msg_len = 0
    while msg_len < n:
      msg_chunk = conn.recv(min(n - msg_len, chunk_size))
      if not msg_chunk:
        return None
      msg_queue.append(bytes(msg_chunk))
      msg_len += len(msg_chunk)

    msg = b''.join(msg_queue)
    if len(msg) != n:
      print_log('recv_large',
          'Error of message "{:s}", size mismatch {:d} != {:d}'.format(msg, len(msg), n))

    return msg

  msg_size = recvall(conn, 4)
  if not msg_size:
    return None
  size = struct.unpack('>I', msg_size)[0]
  json_msg = recvall(conn, size)
  data = json.loads(json_msg.decode('utf-8'))
  return data


def send_large(conn, data):
  json_msg = json.dumps(data)
  msg = bytearray(struct.pack('>I', len(json_msg)))
  msg.extend(json_msg.encode('utf-8'))
  conn.sendall(msg)


def recv_file(conn, filesize, filepath, chunk_size=4096):
  try:
    with open(filepath, 'wb') as f:
      remain = filesize
      while remain > 0:
        recv_data = conn.recv(min(remain, chunk_size))
        if not recv_data:
          return None
        f.write(recv_data)
        remain -= len(recv_data)

  except Exception as e:
    print_log('recv_file',
        'Error occurred: "{:s}"'.format(repr(e)))


def prepare_file(conn, media_root, task_data):
  file_type = task_data['file_type']
  hashcode = task_data['hashcode']
  user_hash = task_data['user']
  file_path = get_file_path(media_root, user_hash, hashcode)
  file_dir = '/'.join(file_path.split('/')[:-1])
  os.system('mkdir -p {}'.format(file_dir))

  if not os.path.exists(file_path):
    tmp_path = file_path + '__downloading__'
    if os.path.exists(tmp_path):
      wait_count = 0
      while os.path.exists(tmp_path):
        if wait_count % 10 == 0:
          print_log('prepare_file',
              'Waiting for downloading {}...'.format(file_path))
        time.sleep(1)
        wait_count += 1
        continue
      time.sleep(1)
      if os.path.exists(file_path):
        print_log('prepare_file',
            'Prepared {}'.format(file_path))
        return file_path

    tmp_f = open(tmp_path, 'wb')
    tmp_f.close()
    print_log('prepare_file',
        'Downloading {}....'.format(file_path))
    req_data = {
      'req_type': 'req_download_file',
      'hash': hashcode,
      'user': user_hash,
    }
    send_large(conn, req_data)
    recv_data = recv_large(conn)
    recv_file(conn, recv_data['file_size'], tmp_path)
    if os.path.exists(tmp_path):
      shutil.move(tmp_path, file_path)
      print_log('prepare_file',
          'Downloaded {}'.format(file_path))
    else:
      print_log('prepare_file',
          'Prepared {}'.format(file_path))

  return file_path


def post_anns(conn, task_data, anns):
  req_data = task_data
  req_data.update({
    'req_type': 'post_ann',
    'anns': anns,
  })
  send_large(conn, req_data)
  _ = recv_large(conn)


def report_progress(conn, task_data, ongoing_frame_id):
  req_data = {
    'req_type': 'set_progress_task',
    'file_type': task_data['file_type'],
    'task_id': task_data['task_id'],
    'user': task_data['user'],
    'ongoing_frame_id': ongoing_frame_id,
  }
  send_large(conn, req_data)
  _ = recv_large(conn)


def report_error(conn, task_data, msg):
  req_data = {
    'req_type': 'error_task',
    'file_type': task_data['file_type'],
    'task_id': task_data['task_id'],
    'user': task_data['user'],
    'msg': msg,
  }
  send_large(conn, req_data)
  _ = recv_large(conn)


def report_finish(conn, task_data):
  req_data = {
    'req_type': 'finish_task',
    'file_type': task_data['file_type'],
    'task_id': task_data['task_id'],
    'user': task_data['user'],
  }
  send_large(conn, req_data)
  _ = recv_large(conn)


def get_video_frames_opencv(file_path, start_frame_id, end_frame_id, label_interval):
  cap = cv2.VideoCapture(file_path)
  cap.set(CV_CAP_PROP_POS_FRAMES, start_frame_id)
  for frame_id in range(start_frame_id, end_frame_id + 1):
    _, img = cap.read()
    if frame_id % label_interval == 0 and img is not None:
      yield (frame_id, img)
  cap.release()


def get_video_frames(file_path, start_frame_id, end_frame_id, label_interval):
  import av
  cap = av.open(file_path)
  for packet in cap.demux():
    frames = packet.decode()
    if frames is None:
      continue

    for frame in frames:
      if frame.__class__ != av.video.frame.VideoFrame:
        continue

      if frame.index < start_frame_id or frame.index > end_frame_id or frame.index % label_interval != 0 or img is None:
        continue

      img = np.asarray(frame.to_image())[:, :, ::-1]
      yield (frame.index, img)

  cap.close()
