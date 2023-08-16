#  lmdeploy 安装和使用

lmdeploy 支持 transformer 结构（例如 LLaMA、LLaMa2、InternLM、Vicuna 等），目前支持 fp16，int8 和 int4。

## 一、安装

安装预编译的 python 包
```
$ python3 -m pip install lmdeploy
```

## 二、fp16 推理

把模型转成 lmdeploy 推理格式，假设 huggingface 版 LLaMa2 模型已下载到 `/models/llama-2-7b-chat` 目录，结果会存到 `workspace` 文件夹

```shell
$ python3 -m lmdeploy.serve.turbomind.deploy llama2 /models/llama-2-7b-chat
```

在命令行中测试聊天效果

```shell
$ python3 -m lmdeploy.turbomind.chat ./workspace
..
double enter to end input >>> who are you

..
Hello! I'm just an AI assistant ..
```

也可以用 gradio 启动 WebUI 来聊天
```shell
python3 -m lmdeploy.serve.gradio.app ./workspace
```

lmdeploy 同样支持原始的 facebook 格式模型、支持 70B 模型分布式推理，用法请查看 [lmdeploy 官方文档](https://github.com/internlm/lmdeploy)。

## 三、kv cache int8 量化

lmdeploy 实现了 kv cache int8 量化，同样的显存可以服务更多并发用户。

首先获取量化参数，结果保存到 fp16 转换好的 `workspace/triton_models/weights` 下，7B 模型也不需要 tensor parallel。 

```shell
$ python3 -m lmdeploy.lite.apis.kv_qparams
  --work_dir /models/llama-2-7b-chat
  --turbomind_dir ./workspace/triton_models/weights
  --kv_sym False
  --num_tp 1
```

然后修改推理配置，开启 kv cache int8。编辑 `workspace/triton_models/weights/config.ini` 
* 把 `use_context_fmha` 改为 0，表示关闭 flashattention
* 把 `quant_policy` 设为 4，表示打开 kv cache 量化

最终执行测试即可
```shell
$ python3 -m lmdeploy.turbomind.chat ./workspace
```

[点击这里](https://github.com/InternLM/lmdeploy/blob/main/docs/zh_cn/quantization.md) 查看 kv cache int8 量化的精度和显存测试报告。

## 四、weight int4 量化

lmdeploy 基于 [AWQ 算法](https://arxiv.org/abs/2306.00978) 实现了 weight int4 量化，相对 fp16 版本，速度是 3.16 倍、显存从 16G 降低到 6.3G。

这里有 AWQ 算法优化好的模型，直接下载。

```shell
$ git clone https://huggingface.co/lmdeploy/llama2-chat-7b-w4
```

执行以下命令，即可在终端与模型对话：

```shell
## 转换模型的layout，存放在默认路径 ./workspace 下
python3 -m lmdeploy.serve.turbomind.deploy \
    --model-name llama2 \
    --model-path ./llama2-chat-7b-w4 \
    --model-format awq \
    --group-size 128

## 推理
python3 -m lmdeploy.turbomind.chat ./workspace
```

[点击这里](https://github.com/InternLM/lmdeploy/blob/main/docs/zh_cn/w4a16.md) 查看 weight int4 量化的显存和速度测试结果。

额外说明，weight int4 和 kv cache int8 二者并不冲突、可以同时打开，节约更多显存。