# prototxt_basic
import sys

def data(txt_file, info):
  txt_file.write('name: "arcface"\n')
  txt_file.write('layer {\n')
  txt_file.write('  name: "data"\n')
  txt_file.write('  type: "Input"\n')
  txt_file.write('  top: "data"\n')
  txt_file.write('  input_param {\n')
#  txt_file.write('    shape: { dim: 10 dim: 3 dim: 224 dim: 224 }\n') # TODO
  txt_file.write('    shape: { dim: 1 dim: 3 dim: 108 dim: 108 }\n') # TODO
  txt_file.write('  }\n')
  txt_file.write('}\n')
  txt_file.write('\n')

def SliceChannel(txt_file, info):
  txt_file.write('layer {\n')
  txt_file.write('  bottom: "%s"\n'       % info['bottom'][0])
  txt_file.write('  top: "%s"\n'          % info['top'])
  txt_file.write('  name: "%s"\n'         % info['top'])
  txt_file.write('  type: "SliceChannel"\n')
  txt_file.write('}\n')
  txt_file.write('\n')

def L2Normalization(txt_file, info):
  txt_file.write('layer {\n')
  txt_file.write('  bottom: "%s"\n'       % info['bottom'][0])
  txt_file.write('  top: "%s"\n'          % info['top'])
  txt_file.write('  name: "%s"\n'         % info['top'])
  txt_file.write('  type: "L2Normalization"\n')
  txt_file.write('}\n')
  txt_file.write('\n')

# ?????
def Cast(txt_file, info):
  pass

def Convolution(txt_file, info):
  #print info
  if info['attrs']['no_bias'] == 'True':
    bias_term = 'false'
  else:
    bias_term = 'true'  
  txt_file.write('layer {\n')
  txt_file.write('	bottom: "%s"\n'       % info['bottom'][0])
  txt_file.write('	top: "%s"\n'          % info['top'])
  txt_file.write('	name: "%s"\n'         % info['top'])
  txt_file.write('	type: "Convolution"\n')
  txt_file.write('	convolution_param {\n')
  txt_file.write('		num_output: %s\n'   % info['attrs']['num_filter'])
  txt_file.write('		kernel_size: %s\n'  % info['attrs']['kernel'].split('(')[1].split(',')[0]) # TODO
  if 'pad' not in info['attrs']:
    print('miss Conv_pad')
    txt_file.write('		pad: %s\n' % 0)  # TODO
  else:
    txt_file.write('		pad: %s\n'          % info['attrs']['pad'].split('(')[1].split(',')[0]) # TODO
#  txt_file.write('		group: %s\n'        % info['attrs']['num_group'])
  txt_file.write('		stride: %s\n'       % info['attrs']['stride'].split('(')[1].split(',')[0])
  txt_file.write('		bias_term: %s\n'    % bias_term)
  txt_file.write('	}\n')
  if 'share' in info.keys() and info['share']:  
    txt_file.write('	param {\n')
    txt_file.write('	  name: "%s"\n'     % info['params'][0])
    txt_file.write('	}\n')
  txt_file.write('}\n')
  txt_file.write('\n')

def ChannelwiseConvolution(txt_file, info):
  Convolution(txt_file, info)
  
def BatchNorm(txt_file, info):
  txt_file.write('layer {\n')
  txt_file.write('  bottom: "%s"\n'       % info['bottom'][0])
  txt_file.write('  top: "%s"\n'          % info['top'])
  txt_file.write('  name: "%s"\n'         % info['top'])
  txt_file.write('  type: "BatchNorm"\n')
  txt_file.write('  batch_norm_param {\n')
  txt_file.write('    use_global_stats: true\n')        # TODO
  txt_file.write('    moving_average_fraction: 0.9\n')  # TODO
  txt_file.write('    eps: 0.00002\n')                    # TODO

  txt_file.write('  }\n')
  txt_file.write('}\n')
  # if info['fix_gamma'] is "False":                    # TODO
  txt_file.write('layer {\n')
  txt_file.write('  bottom: "%s"\n'       % info['top'])
  txt_file.write('  top: "%s"\n'          % info['top'])
  txt_file.write('  name: "%s_scale"\n'   % info['top'])
  txt_file.write('  type: "Scale"\n')
  txt_file.write('  scale_param { bias_term: true }\n')
  txt_file.write('}\n')
  txt_file.write('\n')
  pass


def Activation(txt_file, info):
  txt_file.write('layer {\n')
  txt_file.write('  bottom: "%s"\n'       % info['bottom'][0])
  txt_file.write('  top: "%s"\n'          % info['top'])
  txt_file.write('  name: "%s"\n'         % info['top'])
  if info['attrs']['act_type'] == 'sigmoid':   # TODO
    txt_file.write('  type: "Sigmoid"\n')
  elif info['attrs']['act_type'] == 'relu':
    txt_file.write('  type: "ReLU" \n')
  else:
    print "Unknown avtivate_function" ,info['attrs']['act_type']

  txt_file.write('}\n')
  txt_file.write('\n')
  pass

def Concat(txt_file, info):
  txt_file.write('layer {\n')
  txt_file.write('  name: "%s"\n'         % info['top'])
  txt_file.write('  type: "Concat"\n')
  for bottom_i in info['bottom']:
    txt_file.write('  bottom: "%s"\n'     % bottom_i)
  txt_file.write('  top: "%s"\n'          % info['top'])
  txt_file.write('}\n')
  txt_file.write('\n')
  pass
  
def ElementWiseSum(txt_file, info):
  txt_file.write('layer {\n')
  txt_file.write('  name: "%s"\n'         % info['top'])
  txt_file.write('  type: "Eltwise"\n')
  for bottom_i in info['bottom']:
    txt_file.write('  bottom: "%s"\n'     % bottom_i)
  txt_file.write('  top: "%s"\n'          % info['top'])
  txt_file.write('}\n')
  txt_file.write('\n')
  pass

def Pooling(txt_file, info):
  pool_type = 'AVE' if info['attrs']['pool_type'] == 'avg' else 'MAX'
  txt_file.write('layer {\n')
  txt_file.write('  bottom: "%s"\n'       % info['bottom'][0])
  txt_file.write('  top: "%s"\n'          % info['top'])
  txt_file.write('  name: "%s"\n'         % info['top'])
  txt_file.write('  type: "Pooling"\n')
  txt_file.write('  pooling_param {\n')
  txt_file.write('    pool: %s\n'         % pool_type)       # TODO
  txt_file.write('    kernel_size: %s\n'  % info['attrs']['kernel'].split('(')[1].split(',')[0])
  txt_file.write('    stride: %s\n'       % info['attrs']['stride'].split('(')[1].split(',')[0])
  txt_file.write('    pad: %s\n'          % info['attrs']['pad'].split('(')[1].split(',')[0])
  txt_file.write('  }\n')
  txt_file.write('}\n')
  txt_file.write('\n')
  pass

def Pooling_global(txt_file, info):

  if 'global_pool' not in info['attrs']:
    Pooling(txt_file, info)
    print('miss Pool_global_pool')
    return
  if info['attrs']['global_pool'] == 'False':
     Pooling(txt_file, info)

  elif info['attrs']['global_pool'] == 'True':
    pool_type = 'AVE' if info['attrs']['pool_type'] == 'avg' else 'MAX'
    txt_file.write('layer {\n')
    txt_file.write('  bottom: "%s"\n'     % info['bottom'][0])
    txt_file.write('  top: "%s"\n'        % info['top'])
    txt_file.write('  name: "%s"\n'       % info['top'])
    txt_file.write('  type: "Pooling"\n')
    txt_file.write('  pooling_param {\n')
    txt_file.write('    pool: %s\n'       % pool_type)
    txt_file.write('    global_pooling: true\n')
    txt_file.write('  }\n')
    txt_file.write('}\n')
    txt_file.write('\n')
  pass

def FullyConnected(txt_file, info):
  txt_file.write('layer {\n')
  txt_file.write('  bottom: "%s"\n'     % info['bottom'][0])
  txt_file.write('  top: "%s"\n'        % info['top'])
  txt_file.write('  name: "%s"\n'       % info['top'])
  txt_file.write('  type: "InnerProduct"\n')
  txt_file.write('  inner_product_param {\n')
  txt_file.write('    num_output: %s\n' % info['attrs']['num_hidden'])
  txt_file.write('  }\n')
  txt_file.write('}\n')
  txt_file.write('\n')
  pass

#def Flatten(txt_file, info):
#  pass

def Flatten(txt_file, info):
  txt_file.write('layer {\n')
  txt_file.write('  bottom: "%s"\n'     % info['bottom'][0])
  txt_file.write('  top: "%s"\n'        % info['top'])
  txt_file.write('  name: "%s"\n'       % info['top'])
  txt_file.write('  type: "Flatten"\n')
  txt_file.write('}\n')
  txt_file.write('\n')
  
def SoftmaxOutput(txt_file, info):
  pass


def Reshape(txt_file,info):
  txt_file.write('layer {\n')
  txt_file.write('  bottom: "%s"\n'       % info['bottom'][0])
  txt_file.write('  top: "%s"\n'          % info['top'])
  txt_file.write('  name: "%s"\n'         % info['top'])
  txt_file.write('  type: "Reshape"\n')
  txt_file.write('  reshape_param {\n')
  txt_file.write('    shape {\n')
  txt_file.write('      dim: %s\n'       % info['attrs']['shape'].split('(')[1].split(',')[1])
  txt_file.write('      dim: %s\n'          % info['attrs']['shape'].split('(')[1].split(',')[2])
  txt_file.write('      dim: %s\n' % info['attrs']['shape'].split(')')[0].split(',')[3])
  txt_file.write('    }\n')
  txt_file.write('    axis: 1  \n')
  txt_file.write('  }\n')
  txt_file.write('}\n')
  txt_file.write('\n')


def broadcast_mul(txt_file,info):
  txt_file.write('layer {\n')
  txt_file.write('  name: "%s"\n'         % info['top'])
  txt_file.write('  type: "Broadcastmul"\n')
  for bottom_i in info['bottom']:
    txt_file.write('  bottom: "%s"\n'     % bottom_i)
  txt_file.write('  top: "%s"\n'          % info['top'])
  txt_file.write('}\n')
  txt_file.write('\n')


# ----------------------------------------------------------------
def write_node(txt_file, info):
    if 'label' in info['name']:
        return        
    if info['op'] == 'null' and info['name'] == 'data':
        data(txt_file, info)
    elif info['op'] == 'Convolution':
        Convolution(txt_file, info)
    elif info['op'] == 'ChannelwiseConvolution':
        ChannelwiseConvolution(txt_file, info)
    elif info['op'] == 'BatchNorm':
        BatchNorm(txt_file, info)
    elif info['op'] == 'Activation':
        Activation(txt_file, info)
#    elif info['op'] == 'ElementWiseSum':
    elif info['op'] == 'elemwise_add':
        ElementWiseSum(txt_file, info)
    elif info['op'] == '_Plus':
        ElementWiseSum(txt_file, info)
    elif info['op'] == 'Concat':
        Concat(txt_file, info)
    elif info['op'] == 'Pooling':
#        Pooling(txt_file, info)
        Pooling_global(txt_file, info)
    elif info['op'] == 'Flatten':
        Flatten(txt_file, info)
    elif info['op'] == 'FullyConnected':
        FullyConnected(txt_file, info)
    elif info['op'] == 'SoftmaxOutput':
        SoftmaxOutput(txt_file, info)
###
    elif info['op'] == 'Cast':
        Cast(txt_file, info)
    elif info['op'] == 'SliceChannel':
        SliceChannel(txt_file, info)
    elif info['op'] == 'L2Normalization':
        L2Normalization(txt_file, info)
    elif info['op'] == 'Reshape':
      Reshape(txt_file,info)
    elif info['op'] == 'broadcast_mul':
      broadcast_mul(txt_file,info)

    else:
        print "Unknown mxnet op:", info['op']
        #sys.exit("Warning!  Unknown mxnet op:{}".format(info['op']))





