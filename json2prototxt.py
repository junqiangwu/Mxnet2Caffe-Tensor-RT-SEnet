# -*- coding:utf-8 -*-

import sys
import argparse
import json
from prototxt_basic import *

# single_patch_mxnet/test-symbol.json

parser = argparse.ArgumentParser(description='Convert MXNet jason to Caffe prototxt')
parser.add_argument('--mx-json',     type=str, default='to_be_converted/base-symbol.json')
parser.add_argument('--cf-prototxt', type=str, default='to_be_converted/be-converted.prototxt')

# parser.add_argument('--mx-json',     type=str, default='single_patch_mxnet/test-symbol.json')
# parser.add_argument('--cf-prototxt', type=str, default='single_patch_mxnet/single-test.prototxt')

args = parser.parse_args()

with open(args.mx_json) as json_file:    
  jdata = json.load(json_file)

with open(args.cf_prototxt, "w") as prototxt_file:
  for i_node in range(0,len(jdata['nodes'])):
    #print(i_node,jdata['nodes'][i_node]['name'])
    # node_i  单节点的所有信息
    node_i    = jdata['nodes'][i_node]

    if str(node_i['op']) == 'null' and str(node_i['name']) != 'data':
      continue
    '''
    print('{}, \top:{}, name:{} -> {}'.format(i_node,node_i['op'].ljust(20),
                                        node_i['name'].ljust(30),
                                        node_i['name']).ljust(20))
                                  '''

    ##node[i]个节点  存在的信息 op  name  param  input
    info = node_i
    
    info['top'] = info['name']
    info['bottom'] = []
    info['params'] = []

    for input_idx_i in node_i['inputs']:

      # jdata['nodes'][input_idx_i[0]]  jdana['nodes'][input_index]
      input_i = jdata['nodes'][input_idx_i[0]]

      # 找 bottom
      if str(input_i['op']) != 'null' or (str(input_i['name']) == 'data'):
        info['bottom'].append(str(input_i['name']))

      #
      if str(input_i['op']) == 'null':
        info['params'].append(str(input_i['name']))
        if not str(input_i['name']).startswith(str(node_i['name'])):
          print('           use shared weight -> %s'% str(input_i['name']))
          info['share'] = True
      
    write_node(prototxt_file, info)

