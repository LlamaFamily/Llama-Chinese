## 使用llama.cpp量化部署

以[llama.cpp工具](https://github.com/Rayrtfr/llama.cpp)为例，介绍模型量化并在本地部署的详细步骤。Windows则可能需要cmake等编译工具的安装。**本地快速部署体验推荐使用经过指令精调的[Atom-7B-Chat](https://github.com/LlamaFamily/Llama-Chinese?tab=readme-ov-file#%E5%9F%BA%E4%BA%8Ellama2%E7%9A%84%E4%B8%AD%E6%96%87%E9%A2%84%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8Batom)模型，有条件的推荐使用6-bit或者8-bit模型，效果更佳。** 运行前请确保：

1. 系统应有`make`（MacOS/Linux自带）或`cmake`（Windows需自行安装）编译工具
2. 建议使用Python 3.10以上编译和运行该工具


### Step 1: 克隆和编译llama.cpp

1. （可选）如果已下载旧版仓库，建议`git pull`拉取最新代码，**并执行`make clean`进行清理**
1. 拉取最新版适配过Atom大模型的llama.cpp仓库代码

```bash
$ git clone https://github.com/Rayrtfr/llama.cpp
```

2. 对llama.cpp项目进行编译，生成`./main`（用于推理）和`./quantize`（用于量化）二进制文件。

```bash
$ make
```

**Windows/Linux用户**如需启用GPU推理，则推荐与[BLAS（或cuBLAS如果有GPU）一起编译](https://github.com/Rayrtfr/llama.cpp#blas-build)，可以提高prompt处理速度。以下是和cuBLAS一起编译的命令，适用于NVIDIA相关GPU。参考：[llama.cpp#blas-build](https://github.com/Rayrtfr/llama.cpp#blas-build)

```bash
$ make LLAMA_CUBLAS=1
```

**macOS用户**无需额外操作，llama.cpp已对ARM NEON做优化，并且已自动启用BLAS。M系列芯片推荐使用Metal启用GPU推理，显著提升速度。只需将编译命令改为：`LLAMA_METAL=1 make`，参考[llama.cpp#metal-build](https://github.com/Rayrtfr/llama.cpp#metal-build)

```bash
$ LLAMA_METAL=1 make
```

###  Step 2: 生成量化版本模型

目前llama.cpp已支持`.safetensors`文件以及huggingface格式`.bin`转换为GGUF的FP16格式。

/path/Atom-7B-Chat是模型下载的目录位置。
```bash
$ python convert.py --outfile ./atom-7B-cpp.gguf  /path/Atom-7B-Chat

$ ./quantize ./atom-7B-cpp.gguf ./ggml-atom-7B-q4_0.gguf q4_0
```

### Step 3: 加载并启动模型


- 如果想使用GPU推理：cuBLAS/Metal编译需要指定offload层数，在`./main`中指定例如`-ngl 40`表示offload 40层模型参数到GPU


使用以下命令启动聊天。
```bash
text="<s>Human: 介绍一下北京\n</s><s>Assistant:"
./main -m \
./ggml-atom-7B-q4_0.gguf \
-p "${text}"  \
--logdir ./logtxt 
```
如果要带聊天的上下文，上面的text需要调整成类似这样：
```bash
text="<s>Human: 介绍一下北京\n</s><s>Assistant:北京是一个美丽的城市</s>\n<s>Human: 再介绍一下合肥\n</s><s>Assistant:"
```

更详细的官方说明请参考：[https://github.com/ggerganov/llama.cpp/tree/master/examples/main](https://github.com/ggerganov/llama.cpp/tree/master/examples/main)
