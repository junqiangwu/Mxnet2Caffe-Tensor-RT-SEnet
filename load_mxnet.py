import sys, argparse
import mxnet as mx
try:
    import caffe
except ImportError:
    import os, sys
    #curr_path = os.path.abspath(os.path.dirname(__file__))
#    sys.path.append(os.path.join(curr_path, "/home/chen/Documents/Python/toolbox/caffe-org/python"))
    #sys.path.append(os.path.join(curr_path, "/home/wjq/codes/mx2caffe/caffe/python"))
    sys.path.append("/home/wjq/codes/mx2caffe/caffe/python/")
    import caffe



parser = argparse.ArgumentParser(description='Convert MXNet model to Caffe model')
# #
# parser.add_argument('--mx-model',    type=str, default='single_patch_mxnet/test')
# parser.add_argument('--mx-epoch',    type=int, default=0)
# parser.add_argument('--cf-prototxt', type=str, default='single_patch_caffe/deploy.prototxt')
# parser.add_argument('--cf-model',    type=str, default='single_patch_caffe/arcface.caffemodel')
#
parser.add_argument('--mx-model',    type=str, default='to_be_converted/base')
parser.add_argument('--mx-epoch',    type=int, default=7)
parser.add_argument('--cf-prototxt', type=str, default='to_be_converted/deploy.prototxt')
parser.add_argument('--cf-model',    type=str, default='to_be_converted/convert.caffemodel')
args = parser.parse_args()



_, arg_params, aux_params = mx.model.load_checkpoint('to_be_converted/base',7)

all_keys = arg_params.keys() + aux_params.keys()
all_keys.sort()


#print(all_keys)

for n,i in enumerate(all_keys):
    #print n,i
    if '_weight' in i:
        key_caffe = i.replace('_weight', '')
        #print arg_params[i].asnumpy()
        #print key_caffe,i


net = caffe.Net(args.cf_prototxt, caffe.TRAIN)


for index,i in enumerate(arg_params):
   pass
   #print index,i



net.save(args.cf_model)
print("\n- Finished.\n")
