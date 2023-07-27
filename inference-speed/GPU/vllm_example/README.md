# vllm推理部署

[vllm](https://github.com/vllm-project/vllm)同样是GPU推理的方案。相比较与FasterTrainsformer，vllm更加的简单易用。不需要额外进行模型的转换。支持fp16推理。

特点：

+ 快速的推理速度
+ 高效的kv cache
+ 连续的batch请求推理
+ 优化cuda算子
+ 支持分布式推理

## 第一步： 安装vllm

```bash
git clone https://github.com/vllm-project/vllm

cd vllm && python setup.py install
```

## 第二步：启动测试server

下载[13B模型](https://huggingface.co/FlagAlpha/Llama2-Chinese-13b-Chat/tree/main)或者[7B模型](https://huggingface.co/FlagAlpha/Llama2-Chinese-7b-Chat)到特定目录。

1. 单卡推理

编辑single_gpus_api_server.sh里面model为上面的7B或者13B模型的下载路径。

启动测试server
```bash
# multi_gpus_api_server.sh 里面的CUDA_VISIBLE_DEVICES指定了要使用的GPU卡
bash single_gpus_api_server.sh
```

2. 多卡推理

13B模型，70B模型推荐多卡推理。编辑multi_gpus_api_server.sh里面model为上面的13B模型的下载路径。

启动测试server
```bash
# multi_gpus_api_server.sh 里面的CUDA_VISIBLE_DEVICES指定了要使用的GPU卡
# tensor-parallel-size 指定了卡的个数
bash multi_gpus_api_server.sh
```

## 第三步：启动client测试

```
python client_test.py
```
