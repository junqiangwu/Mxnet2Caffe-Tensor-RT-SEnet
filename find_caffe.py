try:
    import caffe
except ImportError:
    import os, sys
    #curr_path = os.path.abspath(os.path.dirname(__file__))
#    sys.path.append(os.path.join(curr_path, "/home/chen/Documents/Python/toolbox/caffe-org/python"))
    #sys.path.append(os.path.join(curr_path, "/home/wjq/codes/mx2caffe/caffe/python"))
    sys.path.append("/home/wjq/codes/mx2caffe/caffe/python/")
    import caffe

