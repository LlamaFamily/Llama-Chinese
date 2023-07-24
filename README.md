<h1 align="center">
  Llama2-Chinese
</h1>
<p align="center" width="100%">
  <img src="assets/llama.png" alt="Llama" style="width: 20%; display: block; margin: auto;"></a>
</p>
<p align="center">
  <font face="黑体" color=orange size="6"> 最好的中文Llama大模型 </font>
</p>
<p align="center">
  <a href="https://llama.family">在线体验：llama.family</a>
</p>

</br></br>


## 🗂️ 内容导引
- [🐼 国内Llama2最新下载地址上线！](#-国内llama2最新下载地址上线)
- [🔥 社区介绍：Llama2中文社区](#-社区介绍llama2中文社区)
  - [为什么选择Llama2中文社区？](#为什么选择llama2中文社区)
  - [社区活动](#社区活动)
  - [立即加入我们！](#立即加入我们)
- [📢 社区公告](#-社区公告)
    - [2023年7月24日：FlagAlpha新增Llama2-13B中文微调参数！](#2023年7月24日flagalpha新增llama2-13b中文微调参数)
    - [2023年7月24日：llama.family新增Llama2-70B在线体验！](#2023年7月24日llamafamily新增llama2-70b在线体验)
    - [2023年7月23日：Llama2中文微调参数发布至Hugging Face仓库FlagAlpha！](#2023年7月23日llama2中文微调参数发布至hugging-face仓库flagalpha)
    - [2023年7月22日：Llama2在线体验链接llama.family上线，同时包含Meta原版和中文微调版本！](#2023年7月22日llama2在线体验链接llamafamily上线同时包含meta原版和中文微调版本)
    - [2023年7月21日：评测了Meta原始版Llama2 Chat模型的中文问答能力！](#2023年7月21日评测了meta原始版llama2-chat模型的中文问答能力)
    - [2023年7月21日：新增Llama2模型的Hugging Face版本国内下载地址！](#2023年7月21日新增llama2模型的hugging-face版本国内下载地址)
    - [2023年7月20日：新增飞书知识库文档，欢迎大家一起共建！](#2023年7月20日新增飞书知识库文档欢迎大家一起共建)
    - [2023年7月20日：国内Llama2最新下载地址上线！](#2023年7月20日国内llama2最新下载地址上线)
    - [2023年7月19日：正式启动Llama2模型的中文预训练，关注我们获取实时动态！](#2023年7月19日正式启动llama2模型的中文预训练关注我们获取实时动态)
    - [2023年7月19日：Llama2国内下载地址正在启动，敬请期待！](#2023年7月19日llama2国内下载地址正在启动敬请期待)
    - [2023年7月19日：开启Llama2中文社区，欢迎大家加入！](#2023年7月19日开启llama2中文社区欢迎大家加入)
- [📝 数据来源](#-数据来源)
- [⏬ 模型部署](#-模型部署)
  - [预训练模型](#预训练模型)
  - [Chat模型](#chat模型)
  - [模型调用代码示例](#模型调用代码示例)
  - [Gradio快速搭建问答平台](#gradio快速搭建问答平台)
- [💡 模型微调](#-模型微调)
  - [微调过程](#微调过程)
    - [Step1: 环境准备](#step1-环境准备)
    - [Step2: 数据准备](#step2-数据准备)
    - [Step3: 微调脚本](#step3-微调脚本)
  - [中文微调参数](#中文微调参数)
- [🥇 模型评测](#-模型评测)
- [📖 学习资料](#-学习资料)
  - [Meta官方对于Llama2的介绍](#meta官方对于llama2的介绍)
  - [Llama相关论文](#llama相关论文)
  - [Llama2的评测结果](#llama2的评测结果)
- [🎉 致谢](#-致谢)
- [🤔 问题反馈](#-问题反馈)



## 🐼 国内Llama2最新下载地址上线！

- Llama2-7B官网版本：https://pan.xunlei.com/s/VN_kR2fwuJdG1F3CoF33rwpIA1?pwd=z9kf

- Llama2-7B-Chat官网版本：https://pan.xunlei.com/s/VN_kQa1_HBvV-X9QVI6jV2kOA1?pwd=xmra

- Llama2-13B官网版本：https://pan.xunlei.com/s/VN_izibaMDoptluWodzJw4cRA1?pwd=2qqb

- Llama2-13B-Chat官网版本：https://pan.xunlei.com/s/VN_iyyponyapjIDLXJCNfqy7A1?pwd=t3xw

- Llama2-7B Hugging Face版本：https://pan.xunlei.com/s/VN_t0dUikZqOwt-5DZWHuMvqA1?pwd=66ep

- Llama2-7B-Chat Hugging Face版本：https://pan.xunlei.com/s/VN_oaV4BpKFgKLto4KgOhBcaA1?pwd=ufir

- Llama2-13B Hugging Face版本：https://pan.xunlei.com/s/VN_yT_9G8xNOz0SDWQ7Mb_GZA1?pwd=yvgf
  
- Llama2-13B-Chat Hugging Face版本：https://pan.xunlei.com/s/VN_yA-9G34NGL9B79b3OQZZGA1?pwd=xqrg

## 🔥 社区介绍：Llama2中文社区
欢迎来到Llama2中文社区！我们是一个专注于Llama2模型在中文方面的优化和上层建设的高级技术社区。
**\*基于大规模中文数据，从预训练开始对Llama2模型进行中文能力的持续迭代升级\***。
我们热忱欢迎对大模型LLM充满热情的开发者和研究者加入我们的行列。

### 为什么选择Llama2中文社区？
🚀 **高级工程师团队支持**：社区有一批专注为大家服务的NLP高级工程师，我们有着强大的技术支持和丰富的经验，为您提供专业的指导和帮助。

🎯 **中文优化**：我们致力于在Llama2模型的中文处理方面进行优化，探索适用于中文的最佳实践，以提升其性能和适应性。

💡 **创新交流**：我们拥有一支富有创造力和经验的社区成员团队，定期组织线上活动、技术研讨和经验分享，促进成员间的创新交流。

🌐 **全球联结**：我们欢迎来自世界各地的开发者加入社区，构建一个开放、多元化的学习和交流平台。

🤝 **开放共享**：我们鼓励社区成员开源分享代码和模型，推动合作共赢，共同促进中文NLP技术的发展。

### 社区活动
🗓️ **线上讲座**：邀请行业内专家进行线上讲座，分享Llama2在中文NLP领域的最新技术和应用，探讨前沿研究成果。

💻 **项目展示**：成员可展示自己在Llama2中文优化方面的项目成果，获得反馈和建议，促进项目协作。

📚 **学习资源**：社区维护丰富的学习资料库，包括教程、文档和论文解读，为成员提供全面的学习支持。

📝 **论文解读**：社区成员共同解读与Llama2相关的最新研究论文，深入理解前沿算法和方法。

🎉 **主题活动**：定期举办各类主题活动，包括挑战赛、黑客马拉松和技术沙龙，让社区成员在轻松愉快的氛围中交流和学习。

🌟 **奖励计划**：我们设立奖励计划，对社区中积极参与、贡献优秀的成员给予荣誉和奖励，激励更多优秀人才的加入。

📈 **技术咨询**：我们提供技术咨询服务，解答您在Llama2开发和优化过程中遇到的问题，助您快速攻克难关。

🚀 **项目合作**：鼓励成员间的项目合作，共同探索Llama2在实际应用中的潜力，打造创新解决方案。


### 立即加入我们！
📚 **愿景**：无论您是对Llama2已有研究和应用经验的专业开发者，还是对Llama2中文优化感兴趣并希望深入探索的新手，我们都热切期待您的加入。在Llama2中文社区，您将有机会与行业内顶尖人才共同交流，携手推动中文NLP技术的进步，开创更加美好的技术未来！

🔗 **温馨提示**：本社区为专业技术交流平台，我们热切期望志同道合的开发者和研究者加入。请遵守社区准则，共同维护积极向上的学习氛围，任何与Llama2无关的内容和广告将被清理。感谢您的理解和支持！


## 📢 社区公告

#### 2023年7月24日：[FlagAlpha](https://huggingface.co/FlagAlpha)新增Llama2-13B中文微调参数！

#### 2023年7月24日：[llama.family](https://llama.family/)新增Llama2-70B在线体验！

#### 2023年7月23日：Llama2中文微调参数发布至Hugging Face仓库[FlagAlpha](https://huggingface.co/FlagAlpha)！

#### 2023年7月22日：Llama2在线体验链接[llama.family](https://llama.family/)上线，同时包含Meta原版和中文微调版本！

#### 2023年7月21日：评测了Meta原始版Llama2 Chat模型的[中文问答能力](#-模型评测)！

#### 2023年7月21日：新增Llama2模型的Hugging Face版本国内下载地址！

#### 2023年7月20日：新增[飞书知识库文档](https://chinesellama.feishu.cn/wiki/space/7257824476874768388?ccm_open_type=lark_wiki_spaceLink)，欢迎大家一起共建！

#### 2023年7月20日：国内Llama2最新下载地址上线！

#### 2023年7月19日：正式启动Llama2模型的中文预训练，关注我们获取实时动态！

#### 2023年7月19日：Llama2国内下载地址正在启动，敬请期待！

#### 2023年7月19日：开启Llama2中文社区，欢迎大家加入！


## 📝 数据来源

我们计划通过以下数据来优化Llama2的中文能力:

| 类型                                                       | 描述                                                         |
| ---------------------------------------------------------- | ------------------------------------------------------------ |
| 网络数据                                                   | 互联网上公开的网络数据，挑选出去重后的高质量中文数据，涉及到百科、书籍、博客、新闻、公告、小说等高质量长文本数据。 |
| [Wikipedia](https://github.com/goldsmith/Wikipedia)        | 中文Wikipedia的数据                                          |
| [悟道](https://github.com/BAAI-WuDao/Model)                | 中文悟道开源的200G数据                                       |
| [Clue](https://github.com/CLUEbenchmark/CLUEDatasetSearch) | Clue开放的中文预训练数据，进行清洗后的高质量中文长文本数据   |
| 竞赛数据集                                                 | 近年来中文自然语言处理多任务竞赛数据集，约150个              |
| [MNBVC](https://github.com/esbatmop/MNBVC)                 | MNBVC 中清洗出来的部分数据集                                 |

**希望大家如果有较高质量的数据集能够提供给我们，不胜感激!💕💕**



## ⏬ 模型部署

Meta在🤗Hugging Face上提供了所有模型的下载链接：https://huggingface.co/meta-llama


### 预训练模型

Llama2预训练模型包含7B、13B和70B三个版本

| 模型名称   | 🤗模型加载名称             | 下载地址                                                     |
| ---------- | ------------------------- | ------------------------------------------------------------ |
| Llama2-7B  | meta-llama/Llama-2-7b-hf  | [模型下载](https://huggingface.co/meta-llama/Llama-2-7b-hf)  |
| Llama2-13B | meta-llama/Llama-2-13b-hf | [模型下载](https://huggingface.co/meta-llama/Llama-2-13b-hf) |
| Llama2-70B | meta-llama/Llama-2-70b-hf | [模型下载](https://huggingface.co/meta-llama/Llama-2-70b-hf) |

### Chat模型

Llama2-Chat模型基于预训练模型进行了监督微调，具备更强的对话能力

| 模型名称        | 🤗模型加载名称                  | 下载地址                                                     |
| --------------- | ------------------------------ | ------------------------------------------------------------ |
| Llama2-7B-Chat  | meta-llama/Llama-2-7b-chat-hf  | [模型下载](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) |
| Llama2-13B-Chat | meta-llama/Llama-2-13b-chat-hf | [模型下载](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) |
| Llama2-70B-Chat | meta-llama/Llama-2-70b-chat-hf | [模型下载](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf) |


### 模型调用代码示例

```
from transformers import AutoTokenizer, AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-chat-hf',device_map='auto',torch_dtype=torch.float16,load_in_8bit=True)
model =model.eval()
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf',use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
input_ids = tokenizer(['<s>Human: 介绍一下中国\n</s><s>Assistant: '], return_tensors="pt",add_special_tokens=False).input_ids.to('cuda')        
generate_input = {
    "input_ids":input_ids,
    "max_new_tokens":512,
    "do_sample":True,
    "top_k":50,
    "top_p":0.95,
    "temperature":0.3,
    "repetition_penalty":1.3,
    "eos_token_id":tokenizer.eos_token_id,
    "bos_token_id":tokenizer.bos_token_id,
    "pad_token_id":tokenizer.pad_token_id
}
generate_ids  = model.generate(**generate_input)
text = tokenizer.decode(generate_ids[0])
print(text)
```

### Gradio快速搭建问答平台

基于gradio搭建的问答界面，实现了流式的输出，将下面代码复制到控制台运行，以下代码以Llama2-7B-Chat模型为例，<font color="#006600">不同模型只需修改一下代码里的模型名称就好了😊</font><br/>
```
python examples/chat_gradio.py --model_name_or_path meta-llama/Llama-2-7b-chat
```


## 💡 模型微调

本仓库中提供了基于LoRA的微调代码，未来我们将会扩展更多的微调算法，敬请期待！关于LoRA的详细介绍可以参考论文“[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)”以及微软Github仓库[LoRA](https://github.com/microsoft/LoRA)。

### 微调过程

#### Step1: 环境准备

根据[requirements.txt](https://github.com/FlagAlpha/Llama2-Chinese/blob/main/requirements.txt)安装对应的环境依赖。

#### Step2: 数据准备
在data目录下提供了一份用于模型sft的数据样例：
- 训练数据：[data/train_sft.csv](https://github.com/FlagAlpha/Llama2-Chinese/blob/main/data/train_sft.csv)
- 验证数据：[data/dev_sft.csv](https://github.com/FlagAlpha/Llama2-Chinese/blob/main/data/dev_sft.csv)

每个csv文件中包含一列“text”，每一行为一个训练样例，每个训练样例按照以下格式将问题和答案组织为模型输入，您可以按照以下格式自定义训练和验证数据集：
```
"<s>Human: "+问题+"\n</s><s>Assistant: "+答案
```
例如，
```
<s>Human: 用一句话描述地球为什么是独一无二的。</s><s>Assistant: 因为地球是目前为止唯一已知存在生命的行星。</s>
```


#### Step3: 微调脚本

我们提供了用于微调的脚本[train/sft/finetune.sh](https://github.com/FlagAlpha/Llama2-Chinese/blob/main/train/sft/finetune.sh)，通过修改脚本的部分参数实现模型的微调，关于微调的具体代码见[train/sft/finetune_clm_lora.py](https://github.com/FlagAlpha/Llama2-Chinese/blob/main/train/sft/finetune_clm_lora.py)。

### 中文微调参数
我们基于中文指令数据集对Llama2-Chat模型进行了微调，使得Llama2模型有着更强的中文对话能力。LoRA参数以及与基础模型合并的参数均已上传至[Hugging Face](https://huggingface.co/FlagAlpha)，目前包含7B和13B的模型。

| 模型名称   | 🤗模型加载名称             | 基础模型版本 | 下载地址                                                     | 介绍 |
| ---------- | ------------------------- | ------------------------------------------------------------ |  ------------------------- | ------------------------------------------------------------ |
| Llama2-Chinese-7b-Chat-LoRA  | FlagAlpha/Llama2-Chinese-7b-Chat-LoRA  | meta-llama/Llama-2-7b-chat-hf | [模型下载](https://huggingface.co/FlagAlpha/Llama2-Chinese-7b-Chat-LoRA)  |  中文指令微调的LoRA参数 |
| Llama2-Chinese-7b-Chat  | FlagAlpha/Llama2-Chinese-7b-Chat  | meta-llama/Llama-2-7b-chat-hf | [模型下载](https://huggingface.co/FlagAlpha/Llama2-Chinese-7b-Chat)  |  中文指令微调的LoRA参数与基础模型参数合并版本 |
| Llama2-Chinese-13b-Chat-LoRA  | FlagAlpha/Llama2-Chinese-13b-Chat-LoRA  | meta-llama/Llama-2-13b-chat-hf | [模型下载](https://huggingface.co/FlagAlpha/Llama2-Chinese-13b-Chat-LoRA)  |  中文指令微调的LoRA参数 |
| Llama2-Chinese-13b-Chat  | FlagAlpha/Llama2-Chinese-13b-Chat  | meta-llama/Llama-2-13b-chat-hf | [模型下载](https://huggingface.co/FlagAlpha/Llama2-Chinese-13b-Chat)  |  中文指令微调的LoRA参数与基础模型参数合并版本 |

<!-- ## 🚀 未来计划 -->

<!-- ## 💪 增强能力 -->


## 🥇 模型评测
为了能够更加清晰地了解Llama2模型的中文问答能力，我们筛选了一些具有代表性的中文问题，对Llama2模型进行提问。我们测试的模型包含Meta公开的Llama2-7B-Chat和Llama2-13B-Chat两个版本，没有做任何微调和训练。测试问题筛选自[AtomBulb](https://github.com/AtomEcho/AtomBulb)，共95个测试问题，包含：通用知识、语言理解、创作能力、逻辑推理、代码编程、工作技能、使用工具、人格特征八个大的类别。

测试中使用的Prompt如下，例如对于问题“列出5种可以改善睡眠质量的方法”：
```
[INST] 
<<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. The answer always been translate into Chinese language.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.

The answer always been translate into Chinese language.
<</SYS>>

列出5种可以改善睡眠质量的方法
[/INST]
```
Llama2-7B-Chat的测试结果见[meta_eval_7B.md](assets/meta_eval_7B.md)，Llama2-13B-Chat的测试结果见[meta_eval_13B.md](assets/meta_eval_13B.md)。

通过测试我们发现，Meta原始的Llama2 Chat模型对于中文问答的对齐效果一般，大部分情况下都不能给出中文回答，或者是中英文混杂的形式。因此，基于中文数据对Llama2模型进行训练和微调十分必要，我们的中文版Llama2模型也已经在训练中，近期将对社区开放。



## 📖 学习资料
### Meta官方对于[Llama2](https://ai.meta.com/llama)的介绍
自从Meta公司发布第一代LLaMA模型以来，羊驼模型家族繁荣发展。近期Meta发布了Llama2版本，开源可商用，在模型和效果上有了重大更新。Llama2总共公布了7B、13B和70B三种参数大小的模型。相比于LLaMA，Llama2的训练数据达到了2万亿token，上下文长度也由之前的2048升级到4096，可以理解和生成更长的文本。Llama2 Chat模型基于100万人类标记数据微调得到，在英文对话上达到了接近ChatGPT的效果。      

### Llama相关论文
* [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
* [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://scontent-lax3-2.xx.fbcdn.net/v/t39.2365-6/10000000_663429262362723_1696968207443577320_n.pdf?_nc_cat=101&ccb=1-7&_nc_sid=3c67a6&_nc_ohc=5ol-jUSglG4AX9uTu-j&_nc_ht=scontent-lax3-2.xx&oh=00_AfDVmJr77y3bv5GCbJ26w-stMJNXsZPTwVDlWhoIkkb8Lg&oe=64BDB0D1)
### Llama2的评测结果
<p align="center" width="100%">
<img src="https://scontent-lax3-2.xx.fbcdn.net/v/t39.8562-6/361265668_276217774995411_4529778090866658620_n.jpg?_nc_cat=1&ccb=1-7&_nc_sid=6825c5&_nc_ohc=gSMV6flCjbAAX8pE8nm&_nc_ht=scontent-lax3-2.xx&oh=00_AfC53vAix8IkoTlO1Z46g2IfS3p7jb51A8TaIrOK6grRsQ&oe=64BC6826" alt="Llama2Eval" style="width: 100%; display: block; margin: auto;">
</p>


## 🎉 致谢

感谢原子回声[AtomEcho](https://github.com/AtomEcho)团队的技术和资源支持！

感谢 @xzsGenius 对Llama2中文社区的贡献！

感谢 @Z Potentials社区对Llama2中文社区的支持！


## 🤔 问题反馈

如有问题，请在GitHub Issue中提交，在提交问题之前，请先查阅以往的issue是否能解决你的问题。

礼貌地提出问题，构建和谐的讨论社区。

加入[飞书知识库](https://chinesellama.feishu.cn/wiki/space/7257824476874768388?ccm_open_type=lark_wiki_spaceLink)，一起共建社区文档。

加入微信群讨论😍😍

<p align="center" width="100%">
<img src="./assets/wechat.jpeg" alt="Wechat" style="width: 100%; display: block; margin: auto;">
</p>

<p align="center" width="100%">
<img src="https://api.star-history.com/svg?repos=FlagAlpha/Llama2-Chinese&type=Date" alt="Wechat" style="width: 100%; display: block; margin: auto;">
</p>
