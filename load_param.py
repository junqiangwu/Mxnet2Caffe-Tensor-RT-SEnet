import sys, argparse
import mxnet as mx

try:
    import caffe
except ImportError:
    import os, sys
    sys.path.append("/home/wjq/codes/mx2caffe/caffe/python/")
    import caffe


parser = argparse.ArgumentParser(description='Print Caffe-model Param')
# #
parser.add_argument('--mx-model',    type=str, default='single_patch_mxnet/test')
parser.add_argument('--mx-epoch',    type=int, default=0)
parser.add_argument('--cf-prototxt', type=str, default='to_be_converted/be-converted.prototxt')
parser.add_argument('--cf-model',    type=str, default='to_be_converted/be-converted.caffemodel')


args = parser.parse_args()

_, arg_params, aux_params = mx.model.load_checkpoint('single_patch_mxnet/test',0)

all_keys = arg_params.keys() + aux_params.keys()
all_keys.sort()

net = caffe.Net(args.cf_prototxt,args.cf_model, caffe.TRAIN)


# for i,k in enumerate(all_keys):
#     print i,k


# '''
f = open("Converted-test.txt",'w')

for k,v in net.params.items():
    # if 'bn' in k:
    #     if 'scale' in k:
    #         f.write(k)
    #         f.write(str(v[0].data))
    #         f.write(str(v[1].data))
    #         print k, '      ', v[0].data.shape, '      ', v[1].data.shape
    #     else:
    #         f.write(k)
    #         f.write(str(v[0].data))
    #         f.write(str(v[1].data))
    #         print k,'      ',v[0].data.shape,'      ',v[1].data.shape

    if 'scale' in k:
        f.write(k)
        f.write(str(v[0].data))
        f.write(str(v[1].data))
        print k, '      ', v[0].data.shape, '      ', v[1].data.shape


    if 'bn' in k:
        f.write(k)
        f.write(str(v[0].data))
        f.write(str(v[1].data))
        print k, '      ', v[0].data.shape, '      ', v[1].data.shape

    #f.write(k)
    #f.write(str(v[0].data.shape))
    f.write('\n')
    #f.write(str(v[0].data))
    f.write('\n')
    #print k,v[1].data.shape


for n,i in enumerate(all_keys):
    #print n,i
    if '_weight' in i:
        key_caffe = i.replace('_weight', '')
        #print key_caffe,net.params[key_caffe][0].data.shape
        #f.write(key_caffe + '\n')
        #f.write(net.params[key_caffe][0].data + '\n')

        #print arg_params[i].asnumpy()
        #print key_caffe,i

f.close()
# '''


#net.save(args.cf_model)
print("\n- Finished.\n")
