# coding=utf-8
import json
import time
import argparse

import urllib.request

import sys

parser = argparse.ArgumentParser()
parser.add_argument('--model_source', default="llama_chinese", choices =["llama_chinese", "llama2_meta", "llama3_meta"], required=False,type=str)
args = parser.parse_args()

def get_prompt_llama_chinese(
    chat_history, system_prompt=""
) -> str:
    prompt = ''
    for input_text_one in chat_history:
            prompt += "<s>"+input_text_one['role']+": "+input_text_one['content'].strip()+"\n</s>"
    if chat_history[-1]['role']=='Human':
        prompt += "<s>Assistant: "
    else:
        prompt += "<s>Human: "
    prompt = prompt[-2048:]
    if len(system_prompt)>0:
        prompt = '<s>System: '+system_prompt.strip()+'\n</s>'+prompt
                
    return prompt

def get_prompt_llama2_meta(chat_history, system_prompt=""):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    
    sep = " "
    sep2 =" </s><s>"
    stop_token_ids = [2]
    system_template = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
    roles = ("[INST]", "[/INST]")
    seps = [sep, sep2]
    if system_prompt.strip() != "":
        ret = system_template
    else:
        ret = "[INST] "
    for i, chat in enumerate(chat_history):
        message = chat["content"]
        role = chat["role"]
        if message:
            if i == 0:
                ret += message + " "
            else:
                if role == "Human":
                    ret +=  "[INST]" + " " + message + seps[i % 2]
                else:
                    ret +=  "[/INST]" + " " + message + seps[i % 2]
        else:
            if role == "Human":
                ret += "[INST]"
            else:
                ret += "[/INST]"
    print("prompt:{}".format(ret))
    return ret

def get_prompt_llama3_meta(chat_history, system_prompt=""):
    system_format='<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>'
    user_format='<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>'
    assistant_format='<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>\n'
    prompt_str = ''
    # 拼接历史对话
    for item in chat_history:
        if item['role']=='Human':
            prompt_str+=user_format.format(content=item['content'])
        else:
            prompt_str+=assistant_format.format(content=item['content'])
    if len(system_prompt)>0:
        prompt_str = system_format.format(content=system_prompt) + prompt_str
    prompt_str = "<|begin_of_text|>" + prompt_str
    return prompt_str


def test_api_server(chat_history=[], system_prompt=""):
    header = {'Content-Type': 'application/json'}

    if args.model_source == "llama2_meta":
        prompt = get_prompt_llama2_meta(chat_history, system_prompt)
    elif args.model_source == "llama3_meta":
        prompt = get_prompt_llama3_meta(chat_history, system_prompt)
    else:
        prompt = get_prompt_llama_chinese(chat_history, system_prompt)

    data = {
          "prompt": prompt,
          "stream" : False,
          "n" : 1,
          "best_of": 1, 
          "presence_penalty": 0.0, 
          "frequency_penalty": 0.2, 
          "temperature": 0.3, 
          "top_p" : 0.95, 
          "top_k": 50, 
          "use_beam_search": False, 
          "stop": [], 
          "ignore_eos" :False, 
          "max_tokens": 2048, 
          "logprobs": None
    }
    request = urllib.request.Request(
        url='http://127.0.0.1:8090/generate',
        headers=header,
        data=json.dumps(data).encode('utf-8')
    )

    result = None
    try:
        response = urllib.request.urlopen(request, timeout=300)
        res = response.read().decode('utf-8')
        result = json.loads(res)
        print(json.dumps(data, ensure_ascii=False, indent=2))
        print(json.dumps(result, ensure_ascii=False, indent=2))

    except Exception as e:
        print(e)

    return result

if __name__ == "__main__":
    # 多伦对话测试
    """ 多伦对话测试
        last_question = "怎么回来呢"
        inputs = [{"role": "Human", "content": "如何去北京"}, 
                {"role": "Assitant", "content": "乘坐飞机或者轮船"}, 
                {"role" : "Human", "content": last_question}]
    """
    # 单轮对话  
    last_question = "怎么去北京"
    chat_history = [ {"role" : "Human", "content": last_question}]
    test_api_server(chat_history)

