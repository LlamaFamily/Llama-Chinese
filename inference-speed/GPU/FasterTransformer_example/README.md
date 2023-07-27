#  FasterTransformer &&  Triton 安装和使用

FasterTransformer & Triton 加速LLama2模型推理。 目前支持fp16或者Int8推理，Int4目前还不支持。

## 0. 准备环境变量

```bash
export BUILD_DICTIONARY="/data/build"
export TRITON_VERSION=23.04
```


## 一. 镜像构建


1. 构建镜像

```bash
cd $BUILD_DICTIONARY
git clone https://github.com/Rayrtfr/fastertransformer_backend.git 

cd $BUILD_DICTIONARY/fastertransformer_backend

export TRITON_VERSION=23.04

docker build --build-arg TRITON_VERSION=${TRITON_VERSION} -t triton_ft_backend:${TRITON_VERSION}-v-1 -f docker/Dockerfile-1 .


docker build --build-arg TRITON_VERSION=${TRITON_VERSION} -t triton_ft_backend:${TRITON_VERSION} -f docker/Dockerfile-2 .

docker build --build-arg TRITON_VERSION=${TRITON_VERSION} -t triton_ft_backend:${TRITON_VERSION}-final -f docker/Dockerfile-3 .

```
TRITON_VERSION=23.04 这个镜像需的GPU的驱动版本是 Driver Version: 535.54.03，如果你的GPU的驱动不是这个版本，需要[https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/rel-22-12.html#rel-22-12](https://docs.nvidia.com/deeplearning/triton-inference-server/release-notes/rel-22-12.html#rel-22-12)
找到cuda driver 对应版本的 triton-inference-server。



2.启动容器

```
# 启动容器
export TRITON_VERSION=23.04

# 注意需要 BUILD_DICTIONARY 挂载到容器里面
docker run -idt --gpus=all --net=host  --shm-size=4G --name triton_ft_backend_pure \
  -v $BUILD_DICTIONARY:$BUILD_DICTIONARY \
  -p18888:8888 -p18000:8000 -p18001:8001 -p18002:8002 triton_ft_backend:${TRITON_VERSION}-final  bash 

````

## 二.容器内操作

下面介绍一下[Llama2-Chinese-13b-Chat](https://huggingface.co/FlagAlpha/Llama2-Chinese-13b-Chat)模型的权重转换成FasterTransformer格式。 [Llama2-Chinese-7b-Chat](https://huggingface.co/FlagAlpha/Llama2-Chinese-7b-Chat/tree/main)也是类似的方式。

1. 转换权重, 权重转换成FasterTransformer格式

```
cd $BUILD_DICTIONARY

git clone https://github.com/Rayrtfr/FasterTransformer.git

cd $BUILD_DICTIONARY/FasterTransformer

mkdir models && sudo chmod -R 777 ./*

python3 ./examples/cpp/llama/huggingface_llama_convert.py \
-saved_dir=./models/llama \
-in_file=/path/FlagAlpha/Llama2-Chinese-13b-Chat \
-infer_gpu_num=1 \
-weight_data_type=fp16 \
-model_name=llama
```

2. 修改模型配置

- 编辑config.pbtxt

``` bash
mkdir $BUILD_DICTIONARY/triton-model-store/

cd $BUILD_DICTIONARY/triton-model-store/

cp -r $BUILD_DICTIONARY/fastertransformer_backend/all_models/llama $BUILD_DICTIONARY/triton-model-store/

# 修改 triton-model-store/llama/fastertransformer/config.pbtxt

parameters {
  key: "tensor_para_size"
  value: {
    string_value: "1"
  }
}

## 修改 model_checkpoint_path 为上面转换之后的路径
parameters {
  key: "model_checkpoint_path"
  value: {
    string_value: "/ft_workspace/FasterTransformer/models/llama/1-gpu/"
  }
}

## 模型使用int8推理需要加一下面的配置
parameters { 
  key: "int8_mode" 
  value: { 
    string_value: "1"
  } 
}
```


修改 model.py

```
# 修改这两个文件
triton-model-store/llama/preprocess/1/model.py
triton-model-store/llama/postprocess/1/model.py

# 检查 这个路径为tokenier对应的路径
self.tokenizer = LlamaTokenizer.from_pretrained("/path/FlagAlpha/Llama2-Chinese-13b-Chat")
```


3. 编译 FasterTransformer Library

(同一类型的模型，编译一次就行了)
编译之前检查 FasterTransformer/examples/cpp/llama/llama_config.ini

```bash
# 单卡推理这里是1，多卡可以改成卡的数目
tensor_para_size=1

model_dir=/mnt/data/wuyongyu/train/FasterTransformer/models/llama/1-gpu/
```

编译 FasterTransformer
```bash
cd $BUILD_DICTIONARY/FasterTransformer

mkdir build && cd build

git submodule init && git submodule update

pip3 install fire jax jaxlib transformers

cmake -DSM=86 -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DBUILD_MULTI_GPU=ON -D PYTHON_PATH=/usr/bin/python3 ..

make -j12

make install
```


## 三. 启动 triton server

同样在上面的容器内操作。
```
CUDA_VISIBLE_DEVICES=0 /opt/tritonserver/bin/tritonserver  --model-repository=$BUILD_DICTIONARY/triton-model-store/llama/
```
输出
```
I0717 17:17:14.670037 70681 grpc_server.cc:2450] Started GRPCInferenceService at 0.0.0.0:8001
I0717 17:17:14.670495 70681 http_server.cc:3555] Started HTTPService at 0.0.0.0:8000
I0717 17:17:14.713000 70681 http_server.cc:185] Started Metrics Service at 0.0.0.0:8002
```


启动client测试

```
python3 $BUILD_DICTIONARY/fastertransformer_backend/inference_example/llama/llama_grpc_stream_client.py
```
