#  lmdeploy 安装和使用

lmdeploy 支持 transformer 结构（例如 Atom、LLaMA、LLaMa2、InternLM、Vicuna 等），目前支持 fp16，int8 和 int4。

## 一、安装

安装预编译的 python 包
```
python3 -m pip install lmdeploy==0.2.1
```

## 二、转换huggingface模型为lmdeploy格式

把模型转成 lmdeploy 推理格式，假设 huggingface 版 [Atom-7B-Chat](https://huggingface.co/FlagAlpha/Atom-7B-Chat) 模型已下载到 `/models/Atom-7B-Chat` 目录，结果会存到 当前执行命令的`workspace` 文件夹

```shell
lmdeploy convert llama2 /models/Atom-7B-Chat
```
lmdeploy 修改一处bug
```
sed -i 's/from .utils import get_logger/from transformers.utils.logging import get_logger/g' ./workspace/model_repository/preprocessing/1/tokenizer/tokenizer.py
sed -i 's/from .utils import get_logger/from transformers.utils.logging import get_logger/g' ./workspace/model_repository/postprocessing/1/tokenizer/tokenizer.py
```


## 三、kv cache int8 量化
对于最大长度是 2048 的 Atom-7B fp16 模型，服务端每创建 1 个并发，都需要大约 1030MB 显存保存 kv_cache，即便是 A100 80G，能服务的用户也非常有限。
为了降低运行时显存，lmdeploy 实现了 kv cache PTQ 量化，同样的显存可以服务更多并发用户。
首先计算模型参数，保存到临时目录 atom
```shell
mkdir atom
lmdeploy lite calibrate \
  /models/Atom-7B-Chat  \             # huggingface Atom 模型。也支持 llama/vicuna/internlm/baichuan 等
  --calib-dataset 'ptb' \             # 校准数据集，支持 c4, ptb, wikitext2, pileval
  --calib-samples 128   \             # 校准集的样本数，如果显存不够，可以适当调小
  --device 'cuda'       \             # 单条的文本长度，如果显存不够，可以适当调小
  --work-dir atom                     # 保存 pth 格式量化统计参数和量化后权重的文件夹
```
注意：可能需要安装flash_attn
```shell
conda install -c nvidia cuda-nvcc # 为了使用conda内的cuda环境安装 flash_attn
pip install flash_attn
```


然后用 atom 目录里的参数，计算量化参数，保存到转换后参数到 `workspace/triton_models/weights` 下

```shell
lmdeploy lite kv_qparams                 \ 
  ./atom                                 \  # 上一步计算的 atom 结果
  ./workspace/triton_models/weights      \  # 结果保存目录
  --num-tp 1                                # tensor parallel GPU 个数
```

修改推理配置，开启 kv cache int8。编辑 `workspace/triton_models/weights/config.ini` 
* 把 `use_context_fmha` 改为 0，表示关闭 flashattention
* 把 `quant_policy` 设为 4，表示打开 kv cache 量化

最终执行测试即可
```shell
lmdeploy chat turbomind ./workspace
```

[点击这里](https://github.com/InternLM/lmdeploy/blob/main/docs/zh_cn/kv_int8.md) 查看 kv cache int8 量化实现公式、精度和显存测试报告。

## 四、weight int4 量化

lmdeploy 基于 [AWQ 算法](https://arxiv.org/abs/2306.00978) 实现了 weight int4 量化，性能是 FP16 的 2.4 倍以上。显存从 16G 降低到 6.3G。

对于自己的模型，可以用`auto_awq`工具来优化
```shell
# 指定量化导出的模型路径
WORK_DIR="./atom-7b-chta-w4"

lmdeploy lite auto_awq \
$HF_MODEL              \  # huggingface 模型位置
--calib-dataset 'ptb'  \  # 校准数据集，支持 c4, ptb, wikitext2, pileval
--calib-samples 128    \  # 校准集的样本数，如果显存不够，可以适当调小
--calib-seqlen 2048    \  # 单条的文本长度，如果显存不够，可以适当调小  
--w-bits 4             \  # 权重量化的 bit 数
--w-group-size 128     \  # 权重量化分组统计尺寸
--work-dir $WORK_DIR  
```

执行以下命令，启动服务：
```shell
# 这里的路径是上面步骤一中转换模型的layout的输出
FasterTransformer_PATH="/path/workspace"

TP=1
# 指定需要用的显卡
DEVICES="0"
for ((i = 1; i < ${TP}; ++i)); do
    DEVICES="${DEVICES},$i"
done
DEVICES="\"device=${DEVICES}\""

# 在容器内启动服务
docker run -idt \
        --gpus $DEVICES \
        -v $FasterTransformer_PATH:/workspace/models \
        --shm-size 16g \
        -p 33336:22 \
        -p 33337-33400:33337-33400 \
        --cap-add=SYS_PTRACE \
        --cap-add=SYS_ADMIN \
        --security-opt seccomp=unconfined \
        --name lmdeploy \
        --env NCCL_LAUNCH_MODE=GROUP openmmlab/lmdeploy:latest \
        tritonserver \
        --model-repository=/workspace/models/model_repository \
        --allow-http=0 \
        --allow-grpc=1 \
        --grpc-port=33337 \
        --log-verbose=0 \
        --allow-metrics=1
```

客户端测试：
```shell
python test_api_server.py  --tritonserver_addr 127.0.0.1:33337
```

[点击这里](https://github.com/InternLM/lmdeploy/blob/main/docs/zh_cn/w4a16.md) 查看 weight int4 量化的显存和速度测试结果。

额外说明，weight int4 和 kv cache int8 二者并不冲突、可以同时打开，节约更多显存。
