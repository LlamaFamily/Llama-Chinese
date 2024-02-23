# 使用pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel作为基础镜像
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

RUN apt-get update -y --allow-unauthenticated 
RUN apt install -y git vim git-lfs

#设置工作目录
WORKDIR /root/Llama-Chinese

# 从git上克隆llama-chinese仓库
RUN git clone https://github.com/LlamaFamily/Llama-Chinese.git /root/Llama-Chinese 

# tsinghua source
RUN mkdir -p ~/.pip
RUN echo "[global]\nindex-url = https://pypi.tuna.tsinghua.edu.cn/simple" > ~/.pip/pip.conf

# 使用pip安装requirements.txt
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn -r requirements.txt

#克隆Hugging Face仓库
RUN  git clone https://huggingface.co/FlagAlpha/Atom-7B-Chat

#开启7860端口
EXPOSE 7860

#设置启动命令
ENTRYPOINT ["python", "examples/chat_gradio.py", "--model_name_or_path", "/root/Llama-Chinese/Atom-7B-Chat/"]
