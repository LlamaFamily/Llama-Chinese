<p align="left">
    English ｜ <a href="README.md">中文</a>
</p>
<br>

<h1 align="center">
  Llama-Chinese
</h1>
<p align="center" width="100%">
  <img src="assets/llama.png" alt="Llama" style="width: 20%; display: block; margin: auto;"></a>
</p>
<p align="center">
  <font face="黑体" color=orange size="6"> The Best Chinese Llama Large Language Model </font>
</p>

<p align="center">
🤗 <a href="https://huggingface.co/FlagAlpha" target="_blank">Hugging Face</a> • 🤖 <a href="https://www.modelscope.cn/organization/FlagAlpha/" target="_blank">ModelScope</a> • ✡️ <a href="https://wisemodel.cn/models/FlagAlpha/Atom-7B-Chat" target="_blank">WiseModel</a>
</p> 

<p align="center">
  <a href="https://llama.family">Online(Including Llama2, Llama3): llama.family</a>
</p>
<p align="center">
  <a href="https://huggingface.co/FlagAlpha/Atom-7B-Chat">Open-source Chinese Pre-trained LLM Atom based on Llama2</a>
</p>

</br></br>


## 🗂️ Content Guide
- [📌 Chinese Llama Community](#-chinese-llama-community)
  * [🔥 Community Introduction: Chinese Llama Community](#-community-introduction-chinese-llama-community)
  * [📢 Community Announcements](#-community-announcements)
  * [🤗 LLM Model](#-models-downloads)
    + [🤗 Pre-trained Chinese Model Atom based on Llama2](#-atom-models)
    + [🤗 Meta Official Llama2 Model](#meta-official-llama2-models)
    + [🤗 Fine-tuned Chinese Models based on Llama2](#fine-tuned-chinese-models-based-on-llama2)
  * [🌟 Community Source](#-community-source)
    + [GPU Source](#-gpu-source)
    + [Data Source](#-data-source)
    + [Discussion](#-discussion)
    + [Product](#-product)

- [📌 How to use Llama Model?](#-how-to-use-llama-model)
  * [Setup of Llama3](#setup-of-llama3)
  * [Setup of Llama2](#setup-of-llama2)
    + [Simple Setup](#simple-setup)
      - [Simple Setup-Anaconda](#simple-setup-anaconda)
      - [Simple Setup-Docker](#simple-setup-docker)
      - [Simple Setup-llama.cpp](#simple-setup-llamacpp)
      - [Simple Setup-gradio](#simple-setup-gradio)
      - [Simple Setup-API](#simple-setup-api)
    + [🤖 Model Pretraining](#-model-pretraining)
    + [💡 Model Fine-tuning](#-model-fine-tuning)
      - [Step1: Environment Setup](#step1-environment-setup)
      - [Step2: Data Preparation](#step2-data-preparation)
      - [Step3: Fine-tuning Scripts](#step3-fine-tuning-script)
        - [LoRA Fine-tuning](#lora-fine-tuning)
        - [Full-parameter Fine-tuning](#full-parameter-fine-tuning)
      - [Step4: Load Fine-tuned Model](#step4-load-fine-tuned-model)
        - [LoRA Fine-tuning](#lora-fine-tuning-1)
        - [Full-parameter Fine-tuning](#full-parameter-fine-tuning-1)
    + [🍄 Model Quantization](#-model-quantization)
    + [🚀 Inference Acceleration](#-inference-acceleration)
      - [TensorRT-LLM](#tensorrt-llm)
      - [vLLM](#vllm)
      - [JittorLLMs](#jittorllms)
      - [lmdeploy](#lmdeploy)
    + [💪 Extension Capabilities](#-extension-capabilities)
      - [LangChain](#langchain)
  * [🥇 Model Evaluation](#-model-evaluation)
  * [📖 Learning Resources](#-learning-resources)
    + [Llama3](#llama3)
    + [Llama2](#llama2)
      - [Meta Official Introduction to Llama2](#meta-official-introduction-to-llama2)
      - [Llama-related Papers](#llama-related-papers)
      - [Llama2 Evaluation Results](#llama2-evaluation-results)

- [📌 Others](#-others)
  * [🎉 Acknowledgments](#-acknowledgments)
  * [🤔 Issue Feedback](#-issue-feedback)

## 📌 Chinese Llama Community

### 🔥 Community Introduction: Chinese Llama Community

Welcome to the Chinese Llama Community! We are a technical community dedicated to optimizing and building on top of the Llama model for Chinese applications.
**\*Based on large-scale Chinese data, we start pre-training and continuously upgrade the Llama2 model for Chinese capabilities\***.
We warmly welcome developers and researchers passionate about LLM models to join our community.

<details lang="en">

#### Why Choose the Chinese Llama Community?
🚀 **Support from a Team of Senior Engineers**: The community has a team of dedicated NLP senior engineers who provide strong technical support and rich experience to guide and assist you.

🎯 **Chinese Optimization**: We focus on optimizing Llama2 for Chinese processing, exploring the best practices for Chinese to enhance its performance and adaptability.

💡 **Innovative Exchange**: Our community includes a creative and experienced team of members who organize regular online events, technical discussions, and experience sharing to promote innovative exchanges.

🌐 **Global Connectivity**: We welcome developers from around the world to join the community, creating an open and diverse platform for learning and communication.

🤝 **Open Sharing**: We encourage community members to open-source and share code and models, promoting collaborative win-win efforts and advancing the development of Chinese NLP technology.

#### Community Activities
🗓️ **Online Lectures**: Inviting industry experts to conduct online lectures, sharing the latest technology and applications of Llama2 in the Chinese NLP field, and discussing cutting-edge research results.

💻 **Project Showcase**: Members can showcase their project achievements in Llama2 Chinese optimization, receive feedback and suggestions, and promote project collaboration.

📚 **Learning Resources**: The community maintains a rich library of learning materials, including tutorials, documentation, and paper interpretations, providing comprehensive learning support to members.

📝 **Paper Interpretation**: Community members collectively interpret the latest research papers related to Llama2, delving into advanced algorithms and methods.

🎉 **Themed Events**: Regularly organize various themed events, including challenges, hackathons, and technical salons, allowing community members to exchange and learn in a relaxed and enjoyable atmosphere.

🌟 **Reward Program**: We have established a reward program to honor and reward members who actively participate and contribute outstanding work to the community, motivating more outstanding talents to join.

📈 **Technical Consultation**: We provide technical consulting services to answer your questions and help you overcome challenges in the development and optimization of Llama2.

🚀 **Project Collaboration**: Encourage collaboration between members on projects to explore the potential of Llama2 in practical applications and create innovative solutions.

#### Join Us Now!
📚 **Vision**: Whether you are a professional developer or researcher with experience in Llama2 or a newcomer interested in optimizing Llama2 for Chinese, we eagerly look forward to your joining. In the Chinese Llama Community, you will have the opportunity to exchange ideas with top talents in the industry, work together to advance Chinese NLP technology, and create a brighter technological future!

🔗 **Friendly Reminder**: This community is a platform for professional technical exchange. We earnestly hope that like-minded developers and researchers join us. Please adhere to the community guidelines, maintain a positive learning atmosphere, and any content and advertisements unrelated to Llama2 will be removed. Thank you for your understanding and support!

</details>

### 📢 Community Announcements

【Latest】October 8, 2023: Added the inference acceleration feature for JittorLLMs from Tsinghua University [JittorLLMs](#jittorllms)!

【Latest】September 12, 2023: Updated pre-training versions [Atom-7B](https://huggingface.co/FlagAlpha/Atom-7B) and dialogue version [Atom-7B-Chat](https://huggingface.co/FlagAlpha/Atom-7B-Chat) model parameters. The latest Chinese pre-training data size is 100 billion tokens, and the training progress can be viewed at [llama.family](https://llama.family/)!

【Latest】September 2, 2023: Added [pre-training code](#-model-pretraining) and [full-parameter fine-tuning code](#-model-fine-tuning)!

【Latest】August 28, 2023: Released the open-source large model [Atom-7B](https://huggingface.co/FlagAlpha/Atom-7B) based on Llama2 for Chinese pre-training and will continue to be updated. Details can be found in the [community article](https://mp.weixin.qq.com/s/Bdx0JTVh1kgPn5ydYxIkEw)!

【Latest】August 26, 2023: Provided [FastAPI](#fastapi-interface-setup) interface setup script!

【Latest】August 26, 2023: Provided a script to convert Meta official model parameters to a format compatible with Hugging Face [Format Conversion Script](https://github.com/LlamaFamily/Llama-Chinese/blob/main/scripts/convert2hf/README.md)!

【Latest】August 26, 2023: Added [Code Llama](#-code-model) model!

<details lang="en">

- August 15, 2023: Added [PEFT load fine-tuning model parameters](#load-fine-tuned-model) code example!

- August 14, 2023: Launched the [large model data sharing training platform](https://llama.family), allowing everyone to contribute to large model training, even without computing resources. The data contributed by each community member will determine the future capabilities of the model!

- August 3, 2023: Added GPU [inference acceleration](#-inference-acceleration) support for FasterTransformer and vLLM!

- July 31, 2023: 【Major】The first truly meaningful Llama2 Chinese large model is released! Details can be found in the [community article](https://mp.weixin.qq.com/s/lExUU7z_MvgJ7tzQPF8tUQ)

- July 28, 2023: Deployed a Q&A interface through [Docker](#docker-deployment-of-qa-interface)!

- July 27, 2023: Added [LangChain](#langchain) support!

- July 26, 2023: Released a [4-bit quantized compressed version](#-model-quantization) of the Llama2-13B Chinese fine-tuning parameters!

- July 25, 2023: The community's WeChat public account "Llama Chinese Community" is now live. Feel free to follow for the latest updates and dynamics!

- July 24, 2023: [FlagAlpha](https://huggingface.co/FlagAlpha) added Llama2-13B Chinese fine-tuned parameters!

- July 24, 2023: [llama.family](https://llama.family/) added Llama2-70B online experience!

- July 23, 2023: Released Llama2-13B Chinese fine-tuned parameters to the Hugging Face repository [FlagAlpha](https://huggingface.co/FlagAlpha)!

- July 22, 2023: Llama2 online experience link [llama.family](https://llama.family/) is live, including both Meta original and Chinese fine-tuned versions!

- July 21, 2023: Evaluated the Chinese Q&A capability of the Meta original Llama2 Chat model [Model Evaluation](#-model-evaluation)!

- July 21, 2023: Added the Hugging Face version download link for Llama2 models in China!

- July 20, 2023: Added [Feishu Knowledge Base Documentation](https://chinesellama.feishu.cn/wiki/space/7257824476874768388?ccm_open_type=lark_wiki_spaceLink), welcome everyone to contribute!

- July 20, 2023: Chinese Llama2 latest download links are live!

- July 19, 2023: Officially launched the Llama2 Chinese community, stay tuned for real-time updates!

- July 19, 2023: Chinese Llama2 latest download links are in progress, stay tuned!

- July 19, 2023: Launched the Llama2 Chinese community, welcome everyone to join!

</details>

### 🤗 Models Downloads

#### 🔵 Atom Models

The Atom models, created jointly by the Chinese Llama Community and AtomEcho, rank in the top ten of the Chinese Large Language Model Evaluation List C-Eval (submission on August 21).
<p align="center" width="100%">
<img src="./assets/ceval.jpg" alt="ceval" style="width: 100%; display: block; margin: auto;">
</p>


|  Category  | Model Name        | 🤗Model Loading Name                  | Download Link                                                 |
| --------------- | --------------- | ------------------------------ | ------------------------------------------------------------ |
|  Pretrained  | Atom-7B  | FlagAlpha/Atom-7B  | [HuggingFace](https://huggingface.co/FlagAlpha/Atom-7B) \| [ModelScope](https://modelscope.cn/models/FlagAlpha/Atom-7B) \| [WiseModel](https://wisemodel.cn/models/FlagAlpha/Atom-7B) |
|  Chat  | Atom-7B-Chat  | FlagAlpha/Atom-7B-Chat  | [HuggingFace](https://huggingface.co/FlagAlpha/Atom-7B-Chat) \| [ModelScope](https://modelscope.cn/models/FlagAlpha/Atom-7B-Chat) \| [WiseModel](https://wisemodel.cn/models/FlagAlpha/Atom-7B-Chat) |


The Atom series includes Atom-1B, Atom-7B and Atom-13B, with continuous optimization of Chinese language proficiency based on Llama2. Atom-7B and Atom-7B-Chat are fully open source and available for commercial use. You can obtain the models on the [Hugging Face](https://huggingface.co/FlagAlpha) repository. Details are available in [Atom-7B Download](#atom-chinese-pretrained-model-based-on-llama2).

Atom models have the following optimizations for Chinese:

###### Large-scale Chinese Data Pretraining

Atom models are continually pretrained using a large amount of Chinese data, including encyclopedias, books, blogs, news, announcements, novels, financial data, legal data, medical data, code data, professional paper data, and Chinese natural language processing competition datasets. See [📝 Data Sources](#-data-sources) for details.

The massive data is filtered, scored, and deduplicated, resulting in high-quality Chinese data exceeding 1T tokens, continuously added to the training iterations.

###### More Efficient Chinese Vocabulary

To improve the efficiency of Chinese text processing, we optimized the vocabulary of the Llama2 model. First, based on several hundred gigabytes of Chinese text, we expanded the word library to 65,000 words on the basis of the model's vocabulary. Our improvements increased the Chinese encoding/decoding speed by about 350% according to tests. Additionally, we expanded the coverage of the Chinese character set, including all emoji symbols 😊. This makes generating articles with emoji symbols more efficient.

###### Adaptive Context Expansion

Atom large models support a default context of 4K. Through position interpolation (PI) and Neural Tangent Kernel (NTK) methods, the context length can be expanded to 32K after fine-tuning.

###### 📝 Chinese Data

We optimized the Chinese capabilities of Llama2 using the following data:

| Type                                                       | Description                                                   |
| ---------------------------------------------------------- | ------------------------------------------------------------ |
| Web Data                                                   | Publicly available web data on the Internet, selecting deduplicated high-quality Chinese data involving encyclopedias, books, blogs, news, announcements, novels, etc. |
| [Wikipedia](https://github.com/goldsmith/Wikipedia)        | Chinese Wikipedia data                                        |
| [Wudao](https://github.com/BAAI-WuDao/Model)               | 200G of Chinese Wudao open-source data                         |
| [Clue](https://github.com/CLUEbenchmark/CLUEDatasetSearch) | High-quality Chinese long-text data cleaned from Clue's open Chinese pretraining data |
| Competition Datasets                                       | About 150 Chinese natural language processing multi-task competition datasets in recent years |
| [MNBVC](https://github.com/esbatmop/MNBVC)                 | Some datasets cleaned from MNBVC                              |

**If you have high-quality datasets, we would greatly appreciate it if you could provide them to us! 💕💕**


#### Meta Official Llama2 Models

<details lang="en">

The Llama2 pretrained models include 7B, 13B, and 70B versions. The Llama2-Chat model is fine-tuned based on the pretrained models and has enhanced conversational capabilities.

|  Category  | Model Name   | 🤗Model Loading Name             | Download Link                                                 |
|  ----------  | ---------- | ------------------------- | --------------------- |
|  Pretrained  | Llama2-7B  | meta-llama/Llama-2-7b-hf  | [HuggingFace](https://huggingface.co/meta-llama/Llama-2-7b-hf) \| [XunLei](https://pan.xunlei.com/s/VN_t0dUikZqOwt-5DZWHuMvqA1?pwd=66ep) |
|  Pretrained  | Llama2-13B | meta-llama/Llama-2-13b-hf | [HuggingFace](https://huggingface.co/meta-llama/Llama-2-13b-hf) \| [XunLei](https://pan.xunlei.com/s/VN_yT_9G8xNOz0SDWQ7Mb_GZA1?pwd=yvgf) |
|  Pretrained  | Llama2-70B | meta-llama/Llama-2-70b-hf | [HuggingFace](https://huggingface.co/meta-llama/Llama-2-70b-hf) |
|  Chat  | Llama2-7B-Chat  | meta-llama/Llama-2-7b-chat-hf  | [HuggingFace](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) \| [XunLei](https://pan.xunlei.com/s/VN_oaV4BpKFgKLto4KgOhBcaA1?pwd=ufir) |
|  Chat  | Llama2-13B-Chat | meta-llama/Llama-2-13b-chat-hf | [HuggingFace](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) \| [XunLei](https://pan.xunlei.com/s/VN_yA-9G34NGL9B79b3OQZZGA1?pwd=xqrg) |
|  Chat  | Llama2-70B-Chat | meta-llama/Llama-2-70b-chat-hf | [HuggingFace](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf) \| [XunLei](https://pan.xunlei.com/s/VNa_vCGzCy3h3N7oeFXs2W1hA1?pwd=uhxh#) |
| Code  | CodeLlama-7b    |   meta-llama/Llama-2-70b-chat-hf              | [XunLei](https://pan.baidu.com/s/1cIPzdNywWLvQI7_2QanOEQ?pwd=zfwi) |
| Code  | CodeLlama-7b-Python    |   meta-llama/Llama-2-70b-chat-hf              | [XunLei](https://pan.baidu.com/s/1liY8klGoDagYbpw-g-oFag?pwd=i952) |
| Code  | CodeLlama-7b-Instruct    |   meta-llama/Llama-2-70b-chat-hf              | [XunLei](https://pan.baidu.com/s/108o9_DT2E_vfSGtOnDCQVw?pwd=zkt9) |
| Code  | CodeLlama-13b    |   meta-llama/Llama-2-70b-chat-hf              | [XunLei](https://pan.baidu.com/s/1lLaeHv0XEBv0iiZzI1dpnw?pwd=qn99) |
| Code  | CodeLlama-13b-Python    |   meta-llama/Llama-2-70b-chat-hf              | [XunLei](https://pan.baidu.com/s/1OLVfvZS_oqL3oqMKwsI87w?pwd=a78k) |
| Code  | CodeLlama-13b-Instruct    |   meta-llama/Llama-2-70b-chat-hf              | [XunLei](https://pan.baidu.com/s/1HyxJl4w8wElgkZRh2ATrXQ?pwd=seg6) |
| Code  | CodeLlama-34b    |   meta-llama/Llama-2-70b-chat-hf              | [XunLei](https://pan.baidu.com/s/1vEw0pFgIkctPUN4_5_6pIQ?pwd=q8eu) |

Meta officially released Code Llama on August 24, 2023, which is a fine-tuned version of Llama2 based on code data. It provides three versions with different functionalities: Base Model (Code Llama), Python-specific Model (Code Llama - Python), and Instruction-following Model (Code Llama - Instruct), each available in 7B, 13B, and 34B parameter sizes. The capabilities of different models are summarized in the following table:

|  Model Category         |        Model Name         | Code Completion | Code Fill | Instruction Programming |
|-----------------------|------------------------|------|------|------|
| Code Llama            | CodeLlama-7b           | ✅    | ✅    | ❌    |
|                       | CodeLlama-13b          | ✅    | ✅    | ❌    |
|                       | CodeLlama-34b          | ✅    | ❌    | ❌    |
| Code Llama - Python   | CodeLlama-7b-Python    | ✅    | ❌    | ❌    |
|                       | CodeLlama-13b-Python   | ✅    | ❌    | ❌    |
|                       | CodeLlama-34b-Python   | ✅    | ❌    | ❌    |
| Code Llama - Instruct | CodeLlama-7b-Instruct  | ❌    | ✅    | ✅    |
|                       | CodeLlama-13b-Instruct | ❌    | ✅    | ✅    |
|                       | CodeLlama-34b-Instruct | ❌    | ❌    | ✅    |

We provide a [domestic download link for Code Llama](#-latest-downloads-of-llama2-in-china) and an online experience link at [llama.family](https://llama.family/). For detailed information on Code Llama, refer to the official GitHub repository [codellama](https://github.com/facebookresearch/codellama).

</details>

#### Fine-tuned Chinese Models Based on Llama2

We fine-tuned the Llama2-Chat model based on a Chinese instruction dataset, enhancing its Chinese conversational abilities. LoRA parameters and merged parameters with the base model have been uploaded to [Hugging Face](https://huggingface.co/FlagAlpha) and currently include models for 7B and 13B.

|  Category  | Model Name   | 🤗Model Loading Name             | Base Model Version |    Download Link                                                 |
|  ----------  | ---------- | ------------- |  ----------------- | ------------------- |
|  Merged Parameters | Llama2-Chinese-7b-Chat | FlagAlpha/Llama2-Chinese-7b-Chat  |    meta-llama/Llama-2-7b-chat-hf       |[HuggingFace](https://huggingface.co/FlagAlpha/Llama2-Chinese-7b-Chat)  |
|  Merged Parameters | Llama2-Chinese-13b-Chat | FlagAlpha/Llama2-Chinese-13b-Chat|     meta-llama/Llama-2-13b-chat-hf     |[HuggingFace](https://huggingface.co/FlagAlpha/Llama2-Chinese-13b-Chat) |
|  LoRA Parameters | Llama2-Chinese-7b-Chat-LoRA  | FlagAlpha/Llama2-Chinese-7b-Chat-LoRA  |     meta-llama/Llama-2-7b-chat-hf      |[HuggingFace](https://huggingface.co/FlagAlpha/Llama2-Chinese-7b-Chat-LoRA) |
|  LoRA Parameters | Llama2-Chinese-13b-Chat-LoRA | FlagAlpha/Llama2-Chinese-13b-Chat-LoRA |     meta-llama/Llama-2-13b-chat-hf     |[HuggingFace](https://huggingface.co/FlagAlpha/Llama2-Chinese-13b-Chat-LoRA) |


### 🌟 Community Source
The abundance of community resources is a crucial guarantee for community development, covering various aspects including but not limited to the following four: computing power, data, forums, and applications. The positive development and full utilization of these aspects will provide community members with more opportunities and support, driving the entire community towards a more prosperous direction. More details in [llama.family](https://llama.family/).

<details lang="en">
#### 💻 GPU Source

- Provide computing power resources at below-market prices, usable for various computing tasks such as training and inference of deep learning models.
- Offer exclusive online inference services for community members, enabling users to quickly and effectively perform inference operations on models.
- Provide one-click online fine-tuning services, allowing users to conveniently fine-tune models to adapt to different tasks and data.

#### 📊 Data Source
- Open up abundant training data resources covering multiple domains and industries, providing ample data support for model training.
- Provide high-quality, diverse datasets to meet the needs of different users, supporting data sharing and exchange to facilitate the full utilization of data resources.

#### 💬 Discussion
- The community forum provides a platform for community members to engage in online discussions and exchange technical issues.
- On the forum, users can share experiences, ask questions, and provide answers, fostering technical exchange and collaboration.
- The forum can also host regular online events, seminars, etc., to enhance connections and understanding among community members.

#### 📱 Product
- Offer free promotion spaces for showcasing applications, allowing developers to fully present their apps to community members.
- Provide promotional assistance, including but not limited to publicity campaigns, user guidance, etc., to help apps gain more exposure and users.
- Through the community platform, provide collaboration opportunities for outstanding apps, promoting cooperation and communication among app developers, collectively driving the development and growth of applications.

</details>

## 📌 How to use Llama Model?

### Setup of Llama3

### Setup of Llama2

### Simple Setup
You can choose a learning path to start using the Llama series models. It is recommended to use the Chinese pre-trained dialogue model for better support of Chinese language effects.

#### Simple Setup: Anaconda

##### Step 0: Prerequisites
- Make sure that Python version 3.10 or higher is installed.

##### Step 1: Prepare the environment
If you need to set up the environment and install required packages, run the following command.
```bash
git clone https://github.com/LlamaFamily/Llama-Chinese.git
cd Llama-Chinese
pip install -r requirements.txt
```

##### Step 2: Download Model
You can download the Atom-7B-Chat model from the following source.
- [HuggingFace](https://huggingface.co/FlagAlpha)
- [ModelScope](https://modelscope.cn/organization/FlagAlpha)
- [WideModel](https://wisemodel.cn/models/FlagAlpha/Atom-7B-Chat)

###### Step 3：Infer
1. Create a file named quick_start.py and copy the following content into the file.
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
device_map = "cuda:0" if torch.cuda.is_available() else "auto"
model = AutoModelForCausalLM.from_pretrained('FlagAlpha/Atom-7B-Chat',device_map=device_map,torch_dtype=torch.float16,load_in_8bit=True,trust_remote_code=True,use_flash_attention_2=True)
model =model.eval()
tokenizer = AutoTokenizer.from_pretrained('FlagAlpha/Atom-7B-Chat',use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
input_ids = tokenizer(['<s>Human: 介绍一下中国\n</s><s>Assistant: '], return_tensors="pt",add_special_tokens=False).input_ids
if torch.cuda.is_available():
  input_ids = input_ids.to('cuda')
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
2. run quick_start.py 
```bash
python quick_start.py
```


#### Simple Setup: Docker
For details, refer to: [Docker Deployment](https://github.com/LlamaFamily/Llama-Chinese/blob/main/docs/chat_gradio_guide.md)

Step 1: Prepare the Docker image and launch [chat_gradio.py](../examples/chat_gradio.py) through a Docker container.
```bash
git clone https://github.com/LlamaFamily/Llama-Chinese.git

cd Llama-Chinese

docker build -f docker/Dockerfile -t flagalpha/llama2-chinese:gradio .
```

Step 2: Start chat_gradio through Docker-compose.
```bash
cd Llama-Chinese/docker
doker-compose up -d --build
```

#### Simple Setup: llama.cpp
Details: [llama.cpp](https://github.com/LlamaFamily/Llama-Chinese/blob/main/inference-speed/CPU/ggml/README.md)


##### Simple Setup: gradio
Built on Gradio, the Q&A interface implements fluid output. Copy the following code into the console to run. The code below uses the Atom-7B model as an example, <font color="#006600">simply modify the model name in the code for different models 😊</font><br/>

```
python examples/chat_gradio.py --model_name_or_path FlagAlpha/Atom-7B-Chat
```

##### Simple Setup: API
Use FastChat to build an inference service interface consistent with OpenAI.

<details lang="en">

###### step 0：Prerequisites
Install fastchat
```bash
pip3 install "fschat[model_worker,webui]"
```
###### step 1：Run Restful API
Open three terminals and execute the following three commands respectively.

- Fist start controler
```bash
python3 -m fastchat.serve.controller \
--host localhost \
--port 21001
```

- Start Model
```bash
CUDA_VISIBLE_DEVICES="0" python3 -m fastchat.serve.model_worker --model-path /path/Atom-7B-Chat \
--host localhost \
--port 21002 \
--worker-address "http://localhost:21002" \
--limit-worker-concurrency 5 \
--stream-interval 2 \
--gpus "1" \
--load-8bit
```
- Start RESTful API Service
```bash
python3 -m fastchat.serve.openai_api_server \
--host localhost \
--port 21003 \
--controller-address http://localhost:21001
```

###### step 2：test api service
Execute the Python code below to test the API service deployed above.
```python
# coding=utf-8
import json
import time
import urllib.request
import sys
import requests

def test_api_server(input_text):
    header = {'Content-Type': 'application/json'}

    data = {
          "messages": [{"role": "system", "content": ""}, {"role": "user", "content": input_text}],
          "temperature": 0.3, 
          "top_p" : 0.95, 
          "max_tokens": 512, 
          "model": "LLama2-Chinese-13B",
          "stream" : False,
          "n" : 1,
          "best_of": 1, 
          "presence_penalty": 1.2, 
          "frequency_penalty": 0.2,           
          "top_k": 50, 
          "use_beam_search": False, 
          "stop": [], 
          "ignore_eos" :False,
          "logprobs": None
    }
    response = requests.post(
        url='http://127.0.0.1:21003/v1/chat/completions',
        headers=header,
        data=json.dumps(data).encode('utf-8')
    )

    result = None
    try:
        result = json.loads(response.content)
        print(json.dumps(data, ensure_ascii=False, indent=2))
        print(json.dumps(result, ensure_ascii=False, indent=2))

    except Exception as e:
        print(e)

    return result

if __name__ == "__main__":
    test_api_server("如何去北京?")
```
</details>


#### 🤖 Model Pretraining
While the pretraining data for Llama2 has doubled compared to the first generation LLaMA, the proportion of Chinese pretraining data is still very low, accounting for only 0.13%. This results in a relatively weak Chinese capability for the original Llama2. To enhance the model's Chinese capability, two approaches can be adopted: fine-tuning and pretraining.

- Fine-tuning requires fewer computational resources and can quickly create a prototype of a Chinese Llama. However, its drawback is evident – it can only leverage the existing Chinese capabilities of the base model. Due to the limited amount of Chinese training data for Llama2, the potential improvement is also restricted, addressing the symptoms rather than the root cause.

- Pretraining based on large-scale Chinese corpora involves high costs, requiring not only large-scale high-quality Chinese data but also substantial computational resources. However, the advantage is clear – it optimizes the Chinese capability from the model's foundational layers, achieving a fundamental improvement, injecting robust Chinese capabilities into the core of the large model.

We provide the pretraining code for the Llama model to the community, along with [Chinese test data](https://github.com/LlamaFamily/Llama-Chinese/tree/main/data). More data can be found in [Chinese Data](#-chinese-data). The specific code and configurations are as follows:

- Model pretraining script: [train/pretrain/pretrain.sh](https://github.com/LlamaFamily/Llama-Chinese/blob/main/train/pretrain/pretrain.sh)
- Pretraining implementation code: [train/pretrain/pretrain_clm.py](https://github.com/LlamaFamily/Llama-Chinese/blob/main/train/pretrain/pretrain_clm.py)
- [DeepSpeed](https://github.com/microsoft/DeepSpeed) acceleration:
  - For single-card training, ZeRO-2 can be used. See parameters in [train/pretrain/ds_config_zero2.json](https://github.com/LlamaFamily/Llama-Chinese/blob/main/train/pretrain/ds_config_zero2.json).
  - For multi-card training, ZeRO-3 can be used. See parameters in [train/pretrain/ds_config_zero3.json](https://github.com/LlamaFamily/Llama-Chinese/blob/main/train/pretrain/ds_config_zero3.json).
- Training effectiveness metrics: [train/pretrain/accuracy.py](https://github.com/LlamaFamily/Llama-Chinese/blob/main/train/pretrain/accuracy.py)

#### 💡 Model Fine-Tuning

This repository provides both LoRA fine-tuning and full-parameter fine-tuning code. Detailed information about LoRA can be found in the paper "[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)" and the Microsoft GitHub repository [LoRA](https://github.com/microsoft/LoRA).

##### Step1: Environment Setup

Install the necessary environment dependencies according to [requirements.txt](https://github.com/LlamaFamily/Llama-Chinese/blob/main/requirements.txt).

##### Step2: Data Preparation

In the data directory, there is a sample data for the model's SFT:
- Training data: [data/train_sft.csv](https://github.com/LlamaFamily/Llama-Chinese/blob/main/data/train_sft.csv)
- Validation data: [data/dev_sft.csv](https://github.com/LlamaFamily/Llama-Chinese/blob/main/data/dev_sft.csv)

Each CSV file contains a "text" column, with each row representing a training example. Organize questions and answers in the model's input format, as shown below:
```
"<s>Human: "+question+"\n</s><s>Assistant: "+answer
```
For example,
```
<s>Human: Describe why the Earth is unique in one sentence.</s><s>Assistant: Because the Earth is currently the only known planet with existing life.</s>
```

##### Step3: Fine-tuning Scripts

###### LoRA Fine-tuning
LoRA fine-tuning script: [train/sft/finetune_lora.sh](https://github.com/LlamaFamily/Llama-Chinese/blob/main/train/sft/finetune_lora.sh). For details on LoRA fine-tuning implementation, refer to [train/sft/finetune_clm_lora.py](https://github.com/LlamaFamily/Llama-Chinese/blob/main/train/sft/finetune_clm_lora.py). Fine-tuning on a single machine with multiple cards can be achieved by modifying the "--include localhost:0" in the script.

###### Full-parameter Fine-tuning
Full-parameter fine-tuning script: [train/sft/finetune.sh](https://github.com/LlamaFamily/Llama-Chinese/blob/main/train/sft/finetune.sh). For details on full-parameter fine-tuning implementation, refer to [train/sft/finetune_clm.py](https://github.com/LlamaFamily/Llama-Chinese/blob/main/train/sft/finetune_clm.py).

###### Step4: Load Fine-tuned Model

###### LoRA Fine-tuning
For LoRA fine-tuned model parameters, see [Chinese Fine-Tuned Model based on Llama2](#chinese-fine-tuned-model-based-on-llama2). LoRA parameters need to be combined with base model parameters.

Use [PEFT](https://github.com/huggingface/peft) to load both pretraining and fine-tuned model parameters. In the example code below, set "base_model_name_or_path" to the pretraining model's save path and "finetune_model_path" to the fine-tuned model's save path.

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

finetune_model_path = ''  # For example: 'FlagAlpha/Llama2-Chinese-7b-Chat-LoRA'
config = PeftConfig.from_pretrained(finetune_model_path)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
device_map = "cuda:0" if torch.cuda.is_available() else "auto"
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, device_map=device_map, torch_dtype=torch.float16, load_in_8bit=True)
model = PeftModel.from_pretrained(model, finetune_model_path, device_map={"": 0})
model = model.eval()
input_ids = tokenizer(['<s>Human: Introduce Beijing\n</s><s>Assistant: '], return_tensors="pt", add_special_tokens=False).input_ids
if torch.cuda.is_available():
  input_ids = input_ids.to('cuda')
generate_input = {
    "input_ids": input_ids,
    "max_new_tokens": 512,
    "do_sample": True,
    "top_k": 50,
    "top_p": 0.95,
    "temperature": 0.3,
    "repetition_penalty": 1.3,
    "eos_token_id": tokenizer.eos_token_id,
    "bos_token_id": tokenizer.bos_token_id,
    "pad_token_id": tokenizer.pad_token_id
}
generate_ids = model.generate(**generate_input)
text = tokenizer.decode(generate_ids[0])
print(text)
```
###### Full-parameter Fine-tuning
For full-parameter fine-tuned models, use the same calling method as in Model Calling Code Example, just modify the model name or save path accordingly.



#### 🍄 Model Quantization
We have quantized the parameters of the Chinese fine-tuned model to facilitate running with fewer computational resources. Currently, we have uploaded a 4-bit compressed version of the 13B Chinese fine-tuned model [FlagAlpha/Llama2-Chinese-13b-Chat](https://huggingface.co/FlagAlpha/Llama2-Chinese-13b-Chat) as [FlagAlpha/Llama2-Chinese-13b-Chat-4bit](https://huggingface.co/FlagAlpha/Llama2-Chinese-13b-Chat-4bit) on [Hugging Face](https://huggingface.co/FlagAlpha). The specific calling method is as follows:

Environmental requirements:
```
pip install git+https://github.com/PanQiWei/AutoGPTQ.git
```

```python
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM

model = AutoGPTQForCausalLM.from_quantized('FlagAlpha/Llama2-Chinese-13b-Chat-4bit', device="cuda:0")
tokenizer = AutoTokenizer.from_pretrained('FlagAlpha/Llama2-Chinese-13b-Chat-4bit', use_fast=False)
input_ids = tokenizer(['<s>Human: How to land on Mars\n</s><s>Assistant: '], return_tensors="pt", add_special_tokens=False).input_ids.to('cuda')        
generate_input = {
    "input_ids": input_ids,
    "max_new_tokens": 512,
    "do_sample": True,
    "top_k": 50,
    "top_p": 0.95,
    "temperature": 0.3,
    "repetition_penalty": 1.3,
    "eos_token_id": tokenizer.eos_token_id,
    "bos_token_id": tokenizer.bos_token_id,
    "pad_token_id": tokenizer.pad_token_id
}
generate_ids = model.generate(**generate_input)
text = tokenizer.decode(generate_ids[0])
print(text)
```

#### 🚀 Inference Acceleration
As the parameter scale of large models continues to grow, improving model inference speed has become an important research direction with limited computational resources. Common inference acceleration frameworks include lmdeploy, FasterTransformer, vLLM, and JittorLLMs.

##### TensorRT-LLM
[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/tree/main) is developed by NVIDIA, written in C++/CUDA, and supports distributed inference. 

For detailed inference documentation, visit: [inference-speed/GPU/TensorRT-LLM_example](https://github.com/LlamaFamily/Llama-Chinese/tree/main/inference-speed/GPU/TensorRT-LLM_example)

##### vLLM
[vLLM](https://github.com/vllm-project/vllm) is developed by the University of California, Berkeley, with its core technology being PageAttention. It achieves 24 times higher throughput compared to HuggingFace Transformers. Unlike FasterTransformer, vLLM is more user-friendly and does not require additional model conversion. It supports FP16 inference.

For detailed inference documentation, visit: [inference-speed/GPU/vllm_example](https://github.com/LlamaFamily/Llama-Chinese/blob/main/inference-speed/GPU/vllm_example/README.md)

##### JittorLLMs
[JittorLLMs](https://github.com/Jittor/JittorLLMs) is led by Non-ten Technology in collaboration with the Visual Media Research Center at Tsinghua University. It significantly reduces hardware requirements by 80% through a dynamic swap mechanism. Jittor framework, with zero-copy technology, reduces the loading overhead of large models by 40% compared to PyTorch. Moreover, automatic compilation optimization through meta-operators enhances computational performance by over 20%.

For detailed inference documentation, visit: [inference-speed/GPU/JittorLLMs](https://github.com/LlamaFamily/Llama-Chinese/blob/main/inference-speed/GPU/JittorLLMs_example/README.md)

##### lmdeploy
[lmdeploy](https://github.com/InternLM/lmdeploy/) is developed by the Shanghai AI Lab, using C++/CUDA for inference. It provides Python/gRPC/HTTP interfaces and a WebUI for inference, supporting tensor parallel distributed inference and FP16/weight int4/kv cache int8 quantization.

For detailed inference documentation, visit: [inference-speed/GPU/lmdeploy_example](https://github.com/LlamaFamily/Llama-Chinese/tree/main/inference-speed/GPU/lmdeploy_example)

#### 💪 Extension Capabilities

In addition to continually enhancing the intrinsic qualities of large models, such as knowledge base, general understanding, logical reasoning, and imaginative capabilities, we are also actively expanding the extension capabilities of the large models. This includes features like knowledge base retrieval, computational tools, WolframAlpha integration, and software manipulation.

We have initially integrated the LangChain framework to facilitate the development of applications like document retrieval, question-answering bots, and intelligent agents based on the Llama2 model. For more information on LangChain, please refer to [LangChain](https://github.com/langchain-ai/langchain).

##### LangChain
For a simplified implementation using the LangChain framework with the Llama2 LLM class, refer to [examples/llama2_for_langchain.py](https://github.com/LlamaFamily/Llama-Chinese/blob/main/examples/llama2_for_langchain.py). Here is a basic code snippet:

```python
from llama2_for_langchain import Llama2

# Example using FlagAlpha/Atom-7B-Chat
llm = Llama2(model_name_or_path='FlagAlpha/Atom-7B-Chat')

while True:
    human_input = input("Human: ")
    response = llm(human_input)
    print(f"Llama2: {response}")
```


##### 🥇 Model Evaluation

###### Llama3 Evaluation

###### Llama2 Evaluation

<p align="center" width="100%">
<img src="./assets/llama_eval.jpeg" style="width: 100%; display: block; margin: auto;">
</p>

To gain a clearer understanding of the Chinese question-answering capabilities of the Llama2 model, we selected a set of representative Chinese questions for testing. The tested models include Meta's publicly available versions, namely, Llama2-7B-Chat and Llama2-13B-Chat, without any fine-tuning or training. The test questions were curated from [AtomBulb](https://github.com/AtomEcho/AtomBulb), totaling 95 questions covering eight major categories: general knowledge, language understanding, creative ability, logical reasoning, code programming, work skills, tool usage, and personality traits.

The prompt used during testing is as follows, for example, for the question "List 5 methods to improve sleep quality":

```plaintext
[INST] 
<<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. The answer always been translate into Chinese language.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.

The answer always been translate into Chinese language.
<</SYS>>

List 5 methods to improve sleep quality
[/INST]
```
The test results for Llama2-7B-Chat can be found at[meta_eval_7B.md](assets/meta_eval_7B.md)，and for Llama2-13B-Chat at [meta_eval_13B.md](assets/meta_eval_13B.md)。

Through our testing, we observed that Meta's original Llama2 Chat model generally has mediocre alignment with Chinese questions. In most cases, it fails to provide Chinese answers, or the responses are a mixture of Chinese and English. Therefore, it is crucial to train and fine-tune the Llama2 model on Chinese data. Our Chinese version of the Llama2 model is currently undergoing training and will be made available to the community in the near future.


#### 📖 Learning Resources

##### Llama3

##### Llama2
###### Meta Official Introduction to [Llama2](https://ai.meta.com/llama)
Since the release of Meta's first-generation LLaMA model, the Llama model family has thrived. Recently, Meta released the Llama2 version, which is open-source and commercially available, with significant updates in model and performance. Llama2 has models with parameter sizes of 7B, 13B, and 70B. Compared to LLaMA, Llama2's training data has reached 20 trillion tokens, and the context length has been upgraded from the previous 2048 to 4096, allowing it to understand and generate longer text. The Llama2 Chat model, fine-tuned based on 1 million human-labeled data, achieves results close to ChatGPT in English conversations.

###### Llama-related Papers
* [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
* [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)
* [Code Llama: Open Foundation Models for Code](https://ai.meta.com/research/publications/code-llama-open-foundation-models-for-code/)

## 📌 Others

## 🎉 Acknowledgments
Special thanks to the AtomEcho team for their technical and resource support!

Thanks to @xzsGenius for contributions to the Llama2 Chinese community!

Thanks to the Z-Potentials community for supporting the Llama2 Chinese community!

## 🤔 Issue Feedback
If you have any issues, please submit them in the GitHub Issues. Before submitting a new issue, please check existing issues to see if your problem has already been addressed.

Please be polite when raising issues and contribute to building a harmonious discussion community.

Join the [Feishu Knowledge Base](https://chinesellama.feishu.cn/wiki/space/7257824476874768388?ccm_open_type=lark_wiki_spaceLink) to collaboratively build community documentation.

Join the WeChat group for discussions 😍😍


<p align="center" width="100%">
<img src="./assets/wechat.jpeg" alt="Wechat" style="width: 100%; display: block; margin: auto;">
</p>

<p align="center" width="100%">
<img src="https://api.star-history.com/svg?repos=LlamaFamily/Llama-Chinese&type=Date" alt="Wechat" style="width: 100%; display: block; margin: auto;">
</p>
