### MXNet2Caffe: Convert MXNet model to Caffe model,And use Tensor-RT Plugin_Class to improve performance (SEnet)

- caffe_plugin_layer
- mxnet2caffe 
- caffe2TensorRT
- TensorRT_plugin_layer
 
   Provide mxnet to caffe conversion tool,currently supports Conv、BN、Elemwise、Concat、Pooling、Flatten、
Cast、Fully、Slice、L2、Reshape、Broadcast etc. And then use the **TensorRT(4.0)** engine to parse the caffe model 
to improve performance.The project was successfully tested on SEnet.
    
## Code
 * `json2prototxt.py  prototxt_basic.py` Read mxnet_json file and converte to prototxt
 * `mxnet2caffe.py` Read mxnet_model params_dict and converte to .caffemodel
 * `mxnet_test.py` Debug mxnet output and you can compare the result with the converted caffemodel  
 * `load_param.py` Print mxnet or caffe model param_dict
 * `caffe_plugin_layer` Add caffe plugin_layer, but forward computing implementation in Tensorrt_RT Plugin layer
 * `Tensor_RT/Tensor_RT_Plugin.cpp` Add Brocast_layer、Pooling and debug_layer in Tensor_RT by using IPluginExt API


## caffe_plugin_layer
> The caffe framework does't support many Layers_op. If you need to convert a special layer,
 you need to add a layer plugin in the caffe framework and register the Layer_param. 
 Otherwise, you will get an error when building the network structure using this tool.


- Add layer implementations in the caffe/src/caffe/layers/ directory, 
such as broadcast_mul.cpp broadcast_mul.cu,You need to provide forward and reverse operations.
- Add a layer declaration in the caffe/include/caffe/layers/ directory, such as broadcast_mul.hpp
- Register Layer_param in caffe/src/caffe/proto/caffe.proto (read the parameters required for the Layer from prototxt)

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
- Re-make build libcaffe.so
- sudo make pycaffe recompile python interface ***


## mxnet2caffe
- First,you should run json2prototxt.py to convert the structure(json) of the mxnet model to prototxt format.
- Then, run mxnet2caffe.py to read the prototxt network structure build using the API provided by pycaffe.
- Using mxnet_Api Read the param parameter saved by mxnet, and copy the parameter key value pair into the caffemodel file;


## caffe2TensorRT and TensorRT_plugin_layer
> The tensorrt engine can directly parse the caffe model. For unsupported ops, you can manually add them using
 the interface. This project adds a broadcast operation. In addition, it also tests the Pooling_layer，It also adds a test layer, 
 which can separately print the parameters in the structure and assist the Debug.


- Firsy,Inherit the IPluginExt interface to create a custom layer class

- Create a PluginFactory function that will be used to add custom layer classes to the network.

```angular2html

  virtual int getNbOutputs() const = 0;

  virtual Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) = 0;

  virtual void configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, int maxBatchSize) = 0;

  virtual int initialize() = 0;

  virtual void terminate() = 0;

  virtual size_t getWorkspaceSize(int maxBatchSize) const = 0;

  virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) = 0;

  virtual size_t getSerializationSize() = 0;

  virtual void serialize(void* buffer) = 0;

```
This project provide Pooling, broadcast, and test_layer,you can see the implementation in the code.


### TODO:

* ~~Tensor RT supported Se_Resnet~~
