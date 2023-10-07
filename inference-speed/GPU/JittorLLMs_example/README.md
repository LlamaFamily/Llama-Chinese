# JittorLLMs推理部署

## 配置要求

* 内存要求：至少2G，推荐32G
* 显存：可选， 推荐16G
* 操作系统：支持Windows，Mac，Linux全平台。
* 磁盘空间：至少40GB空闲磁盘空间，用于下载参数和存储交换文件。
* Python版本要求至少`3.9`。

磁盘空间不够时，可以通过环境变量`JITTOR_HOME`指定缓存存放路径。
内存或者显存不够，出现进程被杀死的情况，请参考下方，[限制内存消耗的方法](#配置要求低)。

## 部署方法

可以通过下述指令安装依赖。（注意：此脚本会安装Jittor版torch，推荐用户新建环境运行）

```
# 国内使用 gitlink clone
git clone https://gitlink.org.cn/jittor/JittorLLMs.git --depth 1
# github: git clone https://github.com/Jittor/JittorLLMs.git --depth 1
cd JittorLLMs
# -i 指定用jittor的源， -I 强制重装Jittor版torch
pip install -r requirements.txt -i https://pypi.jittor.org/simple -I
```

如果出现找不到jittor版本的错误，可能是您使用的镜像还没有更新，使用如下命令更新最新版：`pip install jittor -U -i https://pypi.org/simple`

部署只需一行命令即可：

```
python cli_demo.py atom7b
```

运行后会自动从服务器上下载模型文件到本地，会占用根目录下一定的硬盘空间。
最开始运行的时候会编译一些CUDA算子，这会花费一些时间进行加载。

内存或者显存不够，出现进程被杀死的情况，请参考下方，[限制内存消耗的方法](#配置要求低)。

### WebDemo

JittorLLM通过gradio库，允许用户在浏览器之中和大模型直接进行对话。

~~~bash
python web_demo.py atom7b
~~~

### 后端服务部署

JittorLLM在api.py文件之中，提供了一个架设后端服务的示例。

~~~bash
python api.py atom7b
~~~

接着可以使用如下代码进行直接访问

~~~python
post_data = json.dumps({'prompt': 'Hello, solve 5x=13'})
print(json.loads(requests.post("http://0.0.0.0:8000", post_data).text)['response'])
~~~

## 配置要求低

针对大模型显存消耗大等痛点，Jittor团队研发了动态交换技术，Jittor框架是世界上首个支持动态图变量自动交换功能的框架，区别于以往的基于静态图交换技术，用户不需要修改任何代码，原生的动态图代码即可直接支持张量交换，张量数据可以在显存-内存-硬盘之间自动交换，降低用户开发难度。

同时，Jittor大模型推理库也是目前对配置门槛要求最低的框架，只需要参数磁盘空间和2G内存，无需显卡，也可以部署大模型，下面是在不同硬件配置条件下的资源消耗与速度对比。可以发现，JittorLLMs在显存充足的情况下，性能优于同类框架，而显存不足甚至没有显卡，JittorLLMs都能以一定速度运行。

节省内存方法，请安装Jittor版本大于1.3.7.8，并添加如下环境变量：
```bash
export JT_SAVE_MEM=1
# 限制cpu最多使用16G
export cpu_mem_limit=16000000000
# 限制device内存（如gpu、tpu等）最多使用8G
export device_mem_limit=8000000000
# windows 用户，请使用powershell
# $env:JT_SAVE_MEM="1"
# $env:cpu_mem_limit="16000000000"
# $env:device_mem_limit="8000000000"
```
用户可以自由设定cpu和设备内存的使用量，如果不希望对内存进行限制，可以设置为`-1`。
```bash
# 限制cpu最多使用16G
export cpu_mem_limit=-1
# 限制device内存（如gpu、tpu等）最多使用8G
export device_mem_limit=-1
# windows 用户，请使用powershell
# $env:JT_SAVE_MEM="1"
# $env:cpu_mem_limit="-1"
# $env:device_mem_limit="-1"
```

如果想要清理磁盘交换文件，可以运行如下命令
```bash
python -m jittor_utils.clean_cache swap
```
