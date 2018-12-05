### MXNet2Caffe: Convert MXNet model to Caffe model,And use Tensor-RT to improve performance
    
   提供了Mxnet常见结构转换为caffe模型，新增了broadcast_mul 和 Reshape,提供的test-convert-model，
 我已经将转换后的caffe模型在Tensor RT中测试，运行正常，后面将在Tensor RT中集成broadcast_mul 和 Reshape
 
 
## Brief Guide

- 首先运行json2prototxt.py，将mxnet模型的结构转换为prototxt格式，
- 然后，运行mxnet2caffe.py 使用pycaffe提供的API读取prototxt的进行网络结构构建，
- 读取mxnet保存的param参数，将参数键值对复制到caffemodel文件中即可；

### Caffe_plugin_layer

> caffe框架对很多Layer不支持，如果你需要转换特殊的层，则需要自己在caffe框架添加层插件，并注册Layer_param，否则在构建网络结构的时候会报错

- 在caffe/src/caffe/layers/  目录下添加层实现   如broadcast_mul.cpp  broadcast_mul.cu（非必须）
- 在caffe/include/caffe/layers/ 目录下添加层申明  如 broadcast_mul.hpp
- 在caffe/src/caffe/proto/caffe.proto 中注册Layer_param(从prototxt中读取Layer所需参数)

```
message LayerParameter {
  optional string name = 1; // the layer name
  optional string type = 2; // the layer type
  repeated string bottom = 3; // the name of each bottom blob
  repeated string top = 4; // the name of each top blob
  ......
  
  optional BroadcastmulParameter broadcastmul_param = 230;
}

message BroadcastmulParameter {  
   
} 
```
  
- 重新make 生成libcaffe.so
- sudo make pycaffe 重新编译python接口(切记)


## Tensor RT

- 因为我要移植到Tensor RT引擎中，所以转换时仅提供了broadcast_mul_layer的空壳函数


 
