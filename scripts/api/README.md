# API 调用

```
您可以选择具体想部署的模型下载
git clone https://huggingface.co/FlagAlpha/Atom-7B-Chat   Atom-7B-Chat
mv Atom-7B-Chat /path/origin_model
```

首先需要安装额外的依赖 `pip install fastapi uvicorn`，然后运行仓库中的 [accelerate_server.py](accelerate_server.py)：

```bash
python accelerate_server.py \
--model_path /path/origin_model \
--gpus "0" \
--infer_dtype "int8" \
--model_source "llama2_chinese"
```
参数说明：
- model_path 模型的本地路径
- gpus 使用的显卡编号，类似"0"、 "0,1"
- infer_dtype 模型加载后的参数数据类型，可以是 int8, float16
- model_source 模型的源，可以是llama2_chinese、llama2_meta、llama3_meta 根据下载的模型不同去区分，如果下载的是[FlagAlpha](https://huggingface.co/FlagAlpha)下载的则用llama2_chinese。


默认部署在本地的 8001 端口，通过 POST 方法进行调用

```bash
python accelerate_client.py
```
