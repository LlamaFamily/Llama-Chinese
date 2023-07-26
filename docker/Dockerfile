# 使用nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04作为基础镜像
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

# 安装Python编译依赖，下载并编译Python，并设置Python 3.10为默认Python版本
RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libncurses5-dev \
    libncursesw5-dev \
    libreadline-dev \
    libsqlite3-dev \
    libgdbm-dev \
    libdb5.3-dev \
    libbz2-dev \
    libexpat1-dev \
    liblzma-dev \
    libffi-dev \
    libgdbm-compat-dev \
    && wget https://www.python.org/ftp/python/3.10.12/Python-3.10.12.tar.xz -O /tmp/Python-3.10.12.tar.xz \
    && tar xvf /tmp/Python-3.10.12.tar.xz -C /tmp \
    && cd /tmp/Python-3.10.12 \
    && ./configure --enable-optimizations --with-ensurepip=install \
    && make -j $(nproc) \
    && make altinstall \
    && update-alternatives --install /usr/bin/python python /usr/local/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip3.10 1 \
    && rm -rf /tmp/Python-3.10.12.tar.xz /tmp/Python-3.10.12 \
    && apt-get install -y git \
    && curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash \
    && apt-get install -y git-lfs \
    && pip install --upgrade pip

#设置工作目录
WORKDIR /root/Llama2-Chinese

# 从git上克隆llama2-chinese仓库
RUN git clone https://github.com/FlagAlpha/Llama2-Chinese.git /root/Llama2-Chinese 

# 使用pip安装requirements.txt
RUN pip install -r requirements.txt

#克隆Hugging Face仓库
RUN  git clone https://huggingface.co/FlagAlpha/Llama2-Chinese-7b-Chat

#开启7860端口
EXPOSE 7860 

#设置启动命令
ENTRYPOINT ["python", "examples/chat_gradio.py", "--model_name_or_path", "/root/Llama2-Chinese/Llama2-Chinese-7b-Chat/"]
