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
pip install vllm
```

## 第二步：启动测试server

从Huggingface下载Atom或者LLama3模型：
```
# 您可以选择具体想部署的模型下载
git clone https://huggingface.co/FlagAlpha/Atom-7B-Chat  Atom-7B-Chat

# 或者下载Meta官方的Llama3模型：
git clone https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct Meta-Llama-3-8B-Instruct
```

1. 单卡推理

编辑single_gpus_api_server.sh里面model为上面模型的下载路径。

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

注意下面的model_source 模型的源，可以是 llama_chinese、llama2_meta、llama3_meta 根据下载的模型不同去区分，如果下载的是[FlagAlpha](https://huggingface.co/FlagAlpha)下载的则用llama_chinese。

```
python client_test.py --model_source llama_chinese
```
