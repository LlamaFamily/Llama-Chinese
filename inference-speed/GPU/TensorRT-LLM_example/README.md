# 使用NVIDIA TensorRT-LLM部署LLama2 或者Atom

[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/tree/main)是NVIDIA开发的高性能推理框架，您可以按照以下步骤来使用TensorRT-LLM部署LLama2模型或者Atom模型。

以下部署流程参考[TensorRT-LLM/example/llama](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/llama)，需要机器Nvidia显卡驱动535版本以上

## Support Matrix
  * FP16
  * FP8
  * INT8 & INT4 Weight-Only
  * SmoothQuant
  * Groupwise quantization (AWQ/GPTQ)
  * FP8 KV CACHE
  * INT8 KV CACHE (+ AWQ/per-channel weight-only)
  * Tensor Parallel
  * STRONGLY TYPED

## 1. 安装TensorRT-LLM
#### 获取TensorRT-LLM代码：

```bash
# TensorRT-LLM 代码需要使用 git-lfs 拉取
apt-get update && apt-get -y install git git-lfs

git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM

# 本流程将使用 v0.7.0 Release 版本
git checkout tags/v0.7.0 -b release/0.7.0
git submodule update --init --recursive
git lfs install
git lfs pull
```
#### 构建docker镜像并安装TensorRT-LLM
```bash
make -C docker release_build
```

#### 运行docker镜像：
```bash
make -C docker release_run
```

## 2. 为LLama2模型构建TensorRT-LLM推理引擎：

#### 进入build文件夹：
```bash
cd ./examples/llama
```

#### 从Huggingface下载Atom或者LLama2模型：
```
# 您可以选择具体想部署的模型下载
git clone https://huggingface.co/FlagAlpha/Atom-7B-Chat      Atom-7B-Chat
mv Atom-7B-Chat /origin_model
```

#### 使用build.py 构建推理引擎：
以下是一个常见事例，更多参数参考[TensorRT-LLM/example/llama](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/llama)
```bash
python build.py --max_batch_size 1 --max_num_tokens 8192  --model_dir /origin_model --dtype float16  --remove_input_padding --use_inflight_batching --paged_kv_cache --use_weight_only --enable_context_fmha --use_gpt_attention_plugin float16  --use_gemm_plugin float16 --output_dir /model/tensorrt_llm/1 --world_size 1 --tp_size 1 --pp_size 1 --max_input_len 7168 --max_output_len 1024 --multi_block_mode --rotary_scaling dynamic 8.0 --rotary_base 500000
```

## 3. 使用TensorRT-LLM Python Runtime进行推理

#### 使用我们提供的python代码类，启动单机单卡服务
```bash
python atom_inference.py \
    /model/tensorrt_llm/1 \   # 第一个参数 build.py 的output路径
    /origin_model \          # 第二个参数模型tokenizer的路径
    如何成为一个更加优秀的人    # 希望问的问题
```