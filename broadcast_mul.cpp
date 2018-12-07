#include <assert.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <sys/stat.h>
#include <time.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <memory>
#include <string.h>
#include <cstdint>

#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "common.h"

#include <sys/time.h>
//#include "plugin.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;

// stuff we know about the network and the caffe input/output blobs
static const int INPUT_H = 108;
static const int INPUT_W = 108;
static const int INPUT_C = 3;
static const int OUTPUT_SIZE = 192;

static Logger gLogger;

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "flat_192";


template<typename T> void write(char*& buffer, const T& val)
{
*reinterpret_cast<T*>(buffer) = val;
buffer += sizeof(T);
}

template<typename T> void read(const char*& buffer, T& val)
{
val = *reinterpret_cast<const T*>(buffer);
buffer += sizeof(T);
}


#if 0
class slice_layer: public IPluginExt
{
private:
    int c,h,w;
    string layer_name;
public:
    slice_layer(const char* name)
        :layer_name(name)
    {
    	printf("construct %s\n",name);
    }

    // create the plugin at runtime from a byte stream
    slice_layer(const char* name,const void* data, size_t length):layer_name(name)
    {
    	printf("construct %s\n",name);
        const char* d = static_cast<const char*>(data), *a = d;

        read(d, c);
        read(d, h);
        read(d, w);
        assert(d == a + length);
    }
    ~slice_layer(){}
    int getNbOutputs() const override
    {
    	printf("getNbOutputs %s\n",layer_name.c_str());
        return 6;
    }

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
    {
        printf("%s getOutputDimensions: index = %d, InputDims = %d \n",layer_name.c_str(),index, nbInputDims);
        printf("output %d %d %d\n",inputs[0].d[0]/6, inputs[0].d[1], inputs[0].d[2]);
        return Dims3(inputs[0].d[0]/6, inputs[0].d[1], inputs[0].d[2]);
    }

    bool supportsFormat(DataType type, PluginFormat format) const override { return (type == DataType::kFLOAT || type == DataType::kHALF) && format == PluginFormat::kNCHW; }

    void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, DataType type, PluginFormat format, int maxBatchSize) override
    {
        assert((type == DataType::kFLOAT || type == DataType::kHALF) && format == PluginFormat::kNCHW);
	c = inputDims[0].d[0];
	h = inputDims[0].d[1];
	w = inputDims[0].d[2];
    }

    int initialize() override
    {
        return 0;
    }

    virtual void terminate() override
    {}

    virtual size_t getWorkspaceSize(int maxBatchSize) const override
    {
        return 0;
    }

    virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override
    {
    //	printf("enqueue %s %d %d %d\n",layer_name.c_str(),c,h,w);
	const float * pbottom = (const float*)inputs[0];
	for(int i = 0; i < 6; i++)
	{
		cudaMemcpy(outputs[i], (const void*)pbottom, sizeof(float) * c * h * w / 6,cudaMemcpyDeviceToDevice);
		pbottom += c*h*w/6;
	}
#if 0
	if(layer_name == "data_slice1")
	{
		float *tmp = (float*) malloc(sizeof(float)*c*h*w);
		cudaMemcpy(tmp, inputs[0], sizeof(float) * c * h * w ,cudaMemcpyDeviceToHost);
		ofstream debug_file("fea_flat.txt");
		for(int i = 0; i < c*h*w; i++)
			debug_file << tmp[i] << endl;
		free(tmp);
		//exit(0);
	}
#endif
        return 0;
    }

    virtual size_t getSerializationSize() override
    {
        return 3*sizeof(int);
    }

    virtual void serialize(void* buffer) override
    {
        char* d = static_cast<char*>(buffer), *a = d;

        write(d, c);
        write(d, h);
        write(d, w);
        assert(d == a + getSerializationSize());
    }
    
};




class normal_layer: public IPluginExt
{   
	string layer_name;
	int c,h,w;
public:
    normal_layer(const char*name):layer_name(name)
    {
    	printf("construct %s\n",name);
    }
    ~normal_layer(){}
    // create the plugin at runtime from a byte stream
    normal_layer(const char*name, const void* data, size_t length)
    :layer_name(name)
    {
    	printf("construct %s\n",name);
        const char* d = static_cast<const char*>(data), *a = d;

        read(d, c);
        read(d, h);
        read(d, w);
        assert(d == a + length);
    }

    int getNbOutputs() const override
    {
    	printf("getNbOutputs %s\n",layer_name.c_str());
        return 1;
    }

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
    {
        printf("normal_layer get dim\n");
        //printf("get dimension %s\n",layer_name.c_str());
        assert(nbInputDims == 1);
        // for(int i = 0; i < nbInputDims; i++)
        // {
        //     printf("l2normalization %d : dim :", inputs[i].nbDims);
        //     for(int j =0 ; j < inputs[i].nbDims; j++)
        //     {
        //         printf("%d\t",inputs[i].d[j]);
        //     }
        //     printf("\n");
        // }
     	return inputs[0];   
    }

    bool supportsFormat(DataType type, PluginFormat format) const override { return (type == DataType::kFLOAT || type == DataType::kHALF) && format == PluginFormat::kNCHW; }

    void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, DataType type, PluginFormat format, int maxBatchSize) override
    {
        assert((type == DataType::kFLOAT || type == DataType::kHALF) && format == PluginFormat::kNCHW);
	c = inputDims[0].d[0];
	h = inputDims[0].d[1];
	w = inputDims[0].d[2];
    }

    int initialize() override
    {
        return 0;
    }

    virtual void terminate() override
    {}

    virtual size_t getWorkspaceSize(int maxBatchSize) const override
    {
        return 0;
    }

    virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override
    {
    //	printf("enqueue %s %d %d %d\n",layer_name.c_str(),c,h,w);
	float *pbottom = (float*)malloc(sizeof(float)*c*h*w);
	cudaMemcpy((void*)pbottom, inputs[0], sizeof(float) * c* h *w, cudaMemcpyDeviceToHost);
#if 0
	if(layer_name == "l2normalization0")
	{
		ofstream fp("l2_0_before");
		for(int i = 0; i < c*h*w; i++)
		{
			fp << pbottom[i] << endl;
		}
	}
#endif
	double sum = 0;
	for(int i = 0; i < c*h*w; i++)
	{
		sum += pbottom[i] * pbottom[i];
	}
	sum = sqrt(sum);
	for(int i = 0; i < c*h*w; i++)
		pbottom[i] /= sum;
	cudaMemcpy(outputs[0], pbottom, c*h*w*sizeof(float), cudaMemcpyHostToDevice);
#if 0
	if(layer_name == "l2normalization0")
	{
		ofstream fp("l2_0_after");
		for(int i = 0; i < c*h*w; i++)
		{
			fp << pbottom[i] << endl;
		}
	}
#endif
	free(pbottom);
        return 0;
    }
    virtual size_t getSerializationSize() override
    {
        return 3*sizeof(int);
    }

    virtual void serialize(void* buffer) override
    {
        char* d = static_cast<char*>(buffer), *a = d;

        write(d, c);
        write(d, h);
        write(d, w);
        assert(d == a + getSerializationSize());
    }
};

enum layer_type{
    SLICE,
    NORMAL,
    UNKNOWN
};
layer_type get_type(const char *name)
{
    if(name[0] == 'd')
        return SLICE;
    else if(name[0] == 'l')
        return NORMAL;
    else
    {
    	return UNKNOWN;
    }
}



// integration for serialization
class PluginFactory : public nvinfer1::IPluginFactory, public nvcaffeparser1::IPluginFactoryExt
{
public:
    // caffe parser plugin implementation
    bool isPlugin(const char* name) override
    {
        return isPluginExt(name);
    }

    bool isPluginExt(const char* name) override
    {
        char tmp[50];
        sprintf(tmp,"%s",name);
        int len = strlen(name);
        tmp[len-1] = 0;
        
        return !strcmp(tmp, "data_slice")||!strcmp(tmp, "l2normalization");
    }

    virtual nvinfer1::IPlugin* createPlugin(const char* layerName, const nvinfer1::Weights* weights, int nbWeights) override
    {
        // there's no way to pass parameters through from the model definition, so we have to define it here explicitly
        layer_type t = get_type(layerName);
        
         if(t == SLICE)
         {
            slice_layer * slice_ptr = new slice_layer(layerName);
   	    slice_ptrs.push_back(slice_ptr);
            return slice_ptr;
         }
         else if(t == NORMAL)
         {
            normal_layer *normal_ptr = new normal_layer(layerName);
		normal_ptrs.push_back(normal_ptr);
            return normal_ptr;
         }
          
        printf("unknown layer! %s\n",layerName);
        return NULL;
    }

    // deserialization plugin implementation
    IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) override
    {
        assert(isPlugin(layerName));
        if(get_type(layerName) == SLICE)
        {
            slice_layer *slice_ptr = new slice_layer(layerName, serialData, serialLength);
		slice_ptrs.push_back(slice_ptr);
            return slice_ptr;
        }
        else if(get_type(layerName) == NORMAL)
        {
            normal_layer *normal_ptr = new normal_layer(layerName,serialData, serialLength);
        	normal_ptrs.push_back(normal_ptr);
            return normal_ptr;
        }
        printf("unknown layer! %s\n",layerName);
	return NULL;
    }

    // User application destroys plugin when it is safe to do so.
    // Should be done after consumers of plugin (like ICudaEngine) are destroyed.
    void destroyPlugin()
    {
	for(auto ptr:slice_ptrs)
		delete ptr;
	for(auto ptr:normal_ptrs)
		delete ptr;
	slice_ptrs.clear();
	normal_ptrs.clear();
    }
    ~PluginFactory()
    {
	destroyPlugin();
    }
    
    std::vector<slice_layer *> slice_ptrs;
    std::vector<normal_layer *> normal_ptrs;
};




class trt_recog{
public:
    trt_recog(char *path)
    {
        cudaSetDevice(0);
        create_engine_from_model(path);
    }
void doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{
    const ICudaEngine& engine = context.getEngine();
    // input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
    // of these, but in this case we know that there is exactly one input and one output.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // note that indices are guaranteed to be less than IEngine::getNbBindings()
    int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME),
        outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // create GPU buffers and a stream
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE*sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // release the stream and the buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}
    ~trt_recog(){
        // Destroy the engine
        context->destroy();
        engine->destroy();
        runtime->destroy();
        // Destroy plugins created by factory
        pluginFactory.destroyPlugin();
    }
    void predict(const float* data, vector<float> &feat)
    {
        feat.resize(OUTPUT_SIZE);
        doInference(data,feat.data(),1);
    }
private:
    void doInference(const float* input, float* output, int batchSize)
    {
        const ICudaEngine& engine = context->getEngine();
        // input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
        // of these, but in this case we know that there is exactly one input and one output.
        assert(engine.getNbBindings() == 2);
        void* buffers[2];

        // In order to bind the buffers, we need to know the names of the input and output tensors.
        // note that indices are guaranteed to be less than IEngine::getNbBindings()
        int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME),
            outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

        // create GPU buffers and a stream
        CHECK(cudaMalloc(&buffers[inputIndex], batchSize * INPUT_H * INPUT_W * sizeof(float)));
        CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

        cudaStream_t stream;
        CHECK(cudaStreamCreate(&stream));

        // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
        CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
        context->enqueue(batchSize, buffers, stream, nullptr);
        CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE*sizeof(float), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);

        // release the stream and the buffers
        cudaStreamDestroy(stream);
        CHECK(cudaFree(buffers[inputIndex]));
        CHECK(cudaFree(buffers[outputIndex]));
    }
    void create_engine_from_model(char* path)
    {
        char *model{nullptr};
        size_t size{0};
        std::ifstream file(path, std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            model = new char[size];
            assert(model);
            file.read(model, size);
            file.close();
        }
        runtime = createInferRuntime(gLogger);
        assert(runtime != nullptr);

        engine = runtime->deserializeCudaEngine(model, size, &pluginFactory);
        assert(engine != nullptr);

        delete []model;

        context = engine->createExecutionContext();
        assert(context != nullptr);
    }
    IRuntime* runtime;
    PluginFactory pluginFactory;
    ICudaEngine* engine;
    IExecutionContext *context;
    Logger gLogger;
};



#endif 


class Broadcast: public IPluginExt{
private:
    int c,h,w;
    string layer_name;
public:
    Broadcast(const char * name):layer_name(name){ 
        printf("construct %s\n",name);
     }
    ~Broadcast(){ }
    
    Broadcast(const char* name,const void* data, size_t length):layer_name(name)
    {
    	printf("deserialize %s\n",name);
        const char* d = static_cast<const char*>(data), *a = d;

        read(d, c);
        read(d, h);
        read(d, w);
        assert(d == a + length);
    }

    int getNbOutputs() const override{
        return 1;
    }

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
    {
        assert(index == 0 && nbInputDims == 2 && inputs[0].nbDims == 3);
        // assert(mNbInputChannels == inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2]);
        // return Dims3(mNbOutputChannels, 1, 1);
        printf("%s getOutputDimensions: index = %d, InputDims = %d %d %d\n",layer_name.c_str(),index, nbInputDims,inputs[0].nbDims,inputs[1].nbDims);
        printf("inputs[0] -> output  %d %d %d \n",inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
        printf("inputs[1] %d %d %d \n",inputs[1].d[0], inputs[1].d[1], inputs[1].d[2]);

        return Dims3(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
    }

    bool supportsFormat(DataType type, PluginFormat format) const override { return (type == DataType::kFLOAT || type == DataType::kHALF) && format == PluginFormat::kNCHW; }

   virtual size_t getWorkspaceSize(int maxBatchSize) const override
    {
        return 0;
    }

    void configureWithFormat(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, DataType type, PluginFormat format, int maxBatchSize) override
    {
        assert((type == DataType::kFLOAT || type == DataType::kHALF) && format == PluginFormat::kNCHW);
        c = inputDims[0].d[0];
	    h = inputDims[0].d[1];
	    w = inputDims[0].d[2];
        
    }

    int initialize() override
    {
        return 0;
    }


    virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override
    {
    	printf("enqueue %s %d %d %d\n",layer_name.c_str(),c,h,w);


        float *pbottom = (float*)malloc(sizeof(float)*c*h*w);
	    cudaMemcpy((void*)pbottom, inputs[0], sizeof(float) * c* h *w, cudaMemcpyDeviceToHost);

        const float * bottom_1 = (const float*)inputs[1];

        // printf("inputs[0] -> output  %d %d %d \n",inputs[0].nbDims);
        // printf("inputs[1] %d %d %d \n",inputs[1].nbDims);

        printf("c w h %d %d %d",c,w,h);

        for(int i=0;i<10;i++){
            printf("pbottom[ %d ] %f",i,pbottom[i]);
        }

        // for(int i=0;i<c;i++){
        //     pbottom[i] = pbottom[i] * bottom_1[i];
        // }

        cudaMemcpy(outputs[0], (const void*)pbottom, sizeof(float) * c * h * w,cudaMemcpyDeviceToDevice);

    // const float * pbottom = (const float*)inputs[0];
	// for(int i = 0; i < 6; i++)
	// {
	// 	cudaMemcpy(outputs[i], (const void*)pbottom, sizeof(float) * c * h * w / 6,cudaMemcpyDeviceToDevice);
	// 	pbottom += c*h*w/6;
	// }

    }
    virtual void terminate() override {}

 
   virtual size_t getSerializationSize() override
    {
        return 3*sizeof(int);
    }

    virtual void serialize(void* buffer) override
    {
        char* d = static_cast<char*>(buffer), *a = d;

        write(d, c);
        write(d, h);
        write(d, w);
        assert(d == a + getSerializationSize());
    }

};


class PluginFactory :public nvinfer1::IPluginFactory, public nvcaffeparser1::IPluginFactoryExt{

public:
    
    std::vector<Broadcast *> broadcast_ptrs;

     bool isPlugin(const char* name) override
    {
        return isPluginExt(name);
    }

    bool isPluginExt(const char* name) override
    {
        return !strncmp(name, "broadcast",9);
    }

    virtual nvinfer1::IPlugin* createPlugin(const char* layerName, const nvinfer1::Weights* weights, int nbWeights) override
    {
        // there's no way to pass parameters through from the model definition, so we have to define it here explicitly
        
        assert(isPlugin(layerName));
        
        Broadcast *broadcast_ptr =new Broadcast(layerName);
        broadcast_ptrs.push_back(broadcast_ptr);
        printf("createPlugin_1_layer_name: %s\n",layerName);
        return broadcast_ptr;
    }

    // deserialization plugin implementation
    IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) override
    {
        assert(isPlugin(layerName));
        
        Broadcast *broadcast_ptr  =new Broadcast(layerName, serialData, serialLength);
        broadcast_ptrs.push_back(broadcast_ptr);

        printf("createPlugin_2_layer_name: %s\n",layerName);
        return broadcast_ptr;
    
    }

    // User application destroys plugin when it is safe to do so.
    // Should be done after consumers of plugin (like ICudaEngine) are destroyed.
    void destroyPlugin()
    {
        
    }

};


void caffeToTRTModel(const std::string& deployFile,                 // name for caffe prototxt
                     const std::string& modelFile,                  // name for model
                     const std::vector<std::string>& outputs,       // network outputs
                     unsigned int maxBatchSize,                     // batch size - NB must be at least as large as the batch we want to run with)
                     nvcaffeparser1::IPluginFactoryExt* pluginFactory, // factory for plugin layers
                     IHostMemory *&trtModelStream)                  // output stream for the TensorRT model
{
    // create the builder
    IBuilder* builder = createInferBuilder(gLogger);

    // parse the caffe model to populate the network, then set the outputs
    INetworkDefinition* network = builder->createNetwork();
    ICaffeParser* parser = createCaffeParser();
    parser->setPluginFactoryExt(pluginFactory);

    // bool fp16 = builder->platformHasFastFp16();
    const IBlobNameToTensor* blobNameToTensor = parser->parse(deployFile.c_str(),
                                                              modelFile.c_str(),
                                                              *network,DataType::kFLOAT);

    // specify which tensors are outputs
    for (auto& s : outputs)
        network->markOutput(*blobNameToTensor->find(s.c_str()));

    // Build the engine
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(10 << 20);
    // builder->setFp16Mode(fp16);

    ICudaEngine* engine = builder->buildCudaEngine(*network);
    assert(engine);

    // we don't need the network any more, and we can destroy the parser
    network->destroy();
    parser->destroy();

    // serialize the engine, then close everything down
    trtModelStream = engine->serialize();

    engine->destroy();
    builder->destroy();
    shutdownProtobufLibrary();
}


void doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{
    const ICudaEngine& engine = context.getEngine();
    // input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
    // of these, but in this case we know that there is exactly one input and one output.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // note that indices are guaranteed to be less than IEngine::getNbBindings()
    int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME),
        outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // create GPU buffers and a stream
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE*sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // release the stream and the buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}


#if 0
void read_img(const char* filename,float* input_data,int h,int w){
    cv::Mat img;

    img = cv::imread(filename, -1);
    // img = cv::imread(filename, cv::IMREAD_COLOR);
    assert(img.empty());
    cv::resize(img,img,cv::Size(h,w));
    float *img_data = (float *)img.data;
   
    unsigned int size=h * w;
    const float pixelMean[3]{ 104.0f, 117.0f, 123.0f }; // also in BGR order

    for(int c=0;c<INPUT_C;c++){
        for (unsigned j = 0; j < size; ++j)
            input_data[c*size + j] = float(img_data[j*INPUT_C + 2 - c]) - pixelMean[c];
    }

}
#endif

int main(int argc, char** argv)
{
    PluginFactory parserPluginFactory;

    IHostMemory *trtModelStream{ nullptr };
    const char * mdole_path = "../../../data/be-converted.caffemodel";
    const char * proto_path = "../../../data/be-converted.prototxt";;
    caffeToTRTModel(proto_path, mdole_path, std::vector < std::string > { OUTPUT_BLOB_NAME }, 1, &parserPluginFactory, trtModelStream);
    
    parserPluginFactory.destroyPlugin();
    assert(trtModelStream != nullptr);

    float *data = (float *)malloc( INPUT_C * INPUT_H * INPUT_W *sizeof(float) );;
    const char * img_file = "/home/wjq/TensorRT-4.0.1.6/data/before_forward.jpg";

    //read_img(img_file,data,108,108);
    
    for (int i = 0; i < 3* INPUT_H * INPUT_W; i++)
        data[i]=1;

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream->data(), trtModelStream->size(), &parserPluginFactory);
    assert(engine != nullptr);
    trtModelStream->destroy();
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);


   // Run inference on input data
    float prob[OUTPUT_SIZE];
    doInference(*context, data, prob, 1);

        // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
    free(data);

    // Print histogram of the output distribution
    std::cout << "\nOutput:\n\n";

    for(int i=0;i<OUTPUT_SIZE;i++){
        printf("%f",prob[i]);
    }

    std::cout << std::endl;

    // deserialize the engine
    // trt_recog recog("model.trt");

    // vector<float> feat;

    // for(int i = 0 ; i < 1000; i++) recog.predict(data,feat);
    // ofstream fout("feat_class");
    // for(auto pt:feat)
    //     fout << pt << endl;

    return 0;
}