# 推理部署

> 训练完之后或者经过微调之后的模型或者直接从[huggingface](https://huggingface.co/FlagAlpha)下载的模型，都需要部署使用。部署也就是指的模型推理，如果直接使用原生的trainsfomers进行部署，速度会比较慢。针对推理有多种加速手段，会带来较快的推理速度。



## 1. GPU推理方案

### 方案一：vllm

[使用说明](../inference-speed/GPU/vllm_example/README.md)

### 方案二：FasterTransformer &&  Triton

[使用说明](../inference-speed/GPU/FasterTransformer_example/README.md)



## 2. CPU 推理方案

### 方案一：llama2.c
[使用说明](../inference-speed/CPU/llama2.c/README.md)
