from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os, sys, types
import json, base64


def import_module(module_path, code):
  package_name = module_path.split('.')[0]
  module_parent = '.'.join(module_path.split('.')[:-1])
  module_name = module_path.split('.')[-1]

  mod = types.ModuleType(module_name.encode('ascii'))
  sys.modules[module_path] = mod
  exec(code, mod.__dict__)

  cmd = 'global {}; {} = mod; '.format(package_name, module_path) if module_parent else 'global {}; '.format(module_path)
  cmd += 'import {}'.format(module_path)
  exec(cmd)


def dump_packages(package_info):
  inlined_packages = {}
  for package in package_info['packages']:
    inlined_code = ' '.join(package['modules']) + '\n'
    for module_name in package['sources']:
      if module_name.__class__ == list:
        module_name, module_path = module_name
      else:
        module_path = os.path.join(package['path'], package['name'], '/'.join(module_name.split('.')) + '.py')
      code = open(module_path, 'r').read()
      inlined_code += '{} {}\n{}\n'.format(module_name, len(code) + 1, code)
    inlined_packages[package['name']] = inlined_code
  return inlined_packages


def load_packages(inlined_packages):
  for package_name, inlined_code in inlined_packages.items():
    import_module(package_name, '')

    end_idx_info = 1
    while inlined_code[end_idx_info] != '\n':
      end_idx_info += 1
    modules = inlined_code[:end_idx_info].split(' ')
    for module in modules:
      import_module('{}.{}'.format(package_name, module), '')
      import_module('{}.{}.__init__'.format(package_name, module), '')

    idx = end_idx_info + 1
    while idx < len(inlined_code):
      start_idx_info = idx
      end_idx_info = idx + 1
      while inlined_code[end_idx_info] != '\n':
        end_idx_info += 1
      module_name, code_len = inlined_code[start_idx_info:end_idx_info].split(' ')
      code_len = int(code_len)
      code = inlined_code[end_idx_info+1:end_idx_info+code_len]
      import_module('{}.{}'.format(package_name, module_name), code)
      idx = end_idx_info + code_len + 1
#__END__


import struct
from Crypto.Cipher import AES
import argparse


def parse_args():
  parser = argparse.ArgumentParser(description='Python package inliner')
  parser.add_argument(
    '--input',
    dest='input',
    help='input json file path',
    default=None,
    type=str
  )
  parser.add_argument(
    '--main',
    dest='main',
    help='key to main routine in json',
    default=None,
    type=str
  )
  parser.add_argument(
    '--debug',
    dest='debug',
    help='debug mode',
    action='store_true'
  )
  parser.add_argument(
    '--make_data',
    dest='make_data',
    help='process data files',
    action='store_true'
  )
  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)
  return parser.parse_args()


def aes_encrypt(aes_key, data):
  data_len = struct.pack('L', len(data))
  padding = os.urandom(16 - ((len(data_len) + len(data)) % 16))
  padded_data = data_len + data + padding
  iv = '\x00' * 16
  aes_engine = AES.new(aes_key, AES.MODE_CBC, iv)
  return aes_engine.encrypt(padded_data)


def aes_decrypt(aes_key, encrypted_data):
  iv = '\x00' * 16
  aes_engine = AES.new(aes_key, AES.MODE_CBC, iv)
  padded_data = aes_engine.decrypt(encrypted_data)
  data_len = struct.unpack('L', padded_data[:8])[0]
  return padded_data[8:8+data_len]


if __name__ == '__main__':
  args = parse_args()

  print('Loading packages...')
  package_info = json.loads(open(args.input, 'r').read())
  inlined_packages = dump_packages(package_info)
  encoded_packages = base64.b64encode(json.dumps(inlined_packages).encode('utf8'))
  main_info = package_info[args.main]

  print('Processing code {}...'.format(main_info['bin']['output']))
  code = open(os.path.abspath(__file__), 'r').read()
  code = code[:code.find('#__END')]

  if args.debug:
    code += 'if __name__ == \'__main__\':\n  load_packages(json.loads(base64.b64decode(\"{}\").decode(\'utf8\')))\n  {}\n'.format(encoded_packages, main_info['bin']['code'])
  else:
    code += 'if __name__ == \'__main__\':\n  try:\n    load_packages(json.loads(base64.b64decode(\"{}\").decode(\'utf8\')))\n    {}\n  except:\n    pass\n'.format(encoded_packages, main_info['bin']['code'])

  with open(main_info['bin']['output'], 'w') as f:
    if 'key' in main_info['bin']:
      f.write(aes_encrypt(main_info['bin']['key'], code.encode('ascii')))
    else:
      f.write(code.encode('ascii'))

  if args.make_data:
    for data_info in main_info['datas']:
      print('Processing data {}...'.format(data_info['output']))
      with open(data_info['output'], 'w') as f:
        if 'key' in data_info:
          f.write(aes_encrypt(data_info['key'], open(data_info['path'], 'rb').read()))
        else:
          f.write(open(data_info['path'], 'rb').read())
