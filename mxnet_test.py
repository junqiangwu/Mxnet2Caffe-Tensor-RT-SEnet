import os,sys
#sys.path.insert(0,'mxnet_0.11')
import mxnet as mx
import time
from mxnet import ndarray as nd
#import matplotlib.pyplot as plt
#import cv2
import numpy as np

np.set_printoptions(threshold='nan')
from collections import namedtuple

Batch = namedtuple('Batch', ['data'])
#fullmodel = mx.sym.load('/mnt/data4/yangling/for_weilong/test.json')
def Normalize(data):
    #m = np.mean(data)
    mx = data.max()
    mn = data.min()
    data = (data - mn) / (mx - mn)
    return data

def load_checkpoint_single(model, param_path):
    #param_name = '%s-%04d.params' % (prefix, epoch)
    arg_params = {}
    aux_params = {}
    save_dict = mx.nd.load(param_path)
    for k, value in save_dict.items():
        arg_type, name = k.split(':', 1)
        if arg_type == 'arg':
            arg_params[name] = value
        if arg_type == 'aux':
            aux_params[name] = value
        else :
            pass
            #raise ValueError("Invalid param file " + param_path)

    model.set_params(arg_params, aux_params, allow_missing=False)
    #model.save_params('test.params') 
    #model.symbol.save('test.json')

    arg_params, aux_params = model.get_params()
    return arg_params, aux_params

def get_image_gray(url):
    np_array = np.ones([1,1,108,108], dtype=float)
    gray_img = nd.array(np_array)

    return gray_img.astype(dtype=np.float32)


# full_param_path = 'to_be_converted/base-0007.params'
# fmodel = mx.sym.load('to_be_converted/base-symbol.json')

full_param_path = 'se_resnet34/base-0000.params'
fmodel = mx.sym.load('se_resnet34/base-symbol.json')


all_layers = fmodel.get_internals()
fmodel = all_layers['flat_output']

#fullmodel = mx.mod.Module(symbol=fmodel,context=[mx.cpu(0)],data_names=['data'],label_names=[])
fullmodel = mx.mod.Module(symbol=fmodel,data_names=['data'],label_names=[])

img = []

img = get_image_gray('before_forward.jpg')
fullmodel.bind(data_shapes=[('data', (1, 1, 108, 108))], label_shapes=None, for_training=False, force_rebind=False)


arg_params, aux_params = load_checkpoint_single(fullmodel, full_param_path)
fullmodel.set_params(arg_params,aux_params)


file1=open('se_resnet34.txt','w')

tic=time.time()

fullmodel.forward(Batch([mx.nd.array(img)]))

print(time.time()-tic)


prob = fullmodel.get_outputs()[0].asnumpy()
#for feat in prob:
#	feat = normalize(feat)
prob = prob.astype(np.float64)
prob = prob.reshape(-1,1)
np.savetxt(file1,prob,fmt='%.12f')



file1.close()
