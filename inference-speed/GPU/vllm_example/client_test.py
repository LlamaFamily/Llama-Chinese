# coding=utf-8
import json
import time

import urllib.request

import sys


def gen_prompt(input_text):
    prompt =  "<s>Human: "+input_text+"\n</s><s>Assistant: "
    return prompt
    

def test_api_server(input_text):
    header = {'Content-Type': 'application/json'}

    prompt = gen_prompt(input_text.strip())

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
    test_api_server("如何去北京?")
    test_api_server("肚子疼怎么办?")
    test_api_server("帮我写一个请假条")
