# coding=utf-8
import json
import time
import urllib.request
import sys

def test_api_server(input_text):
    header = {'Content-Type': 'application/json'}

    data = {
          "system_prompt": "",
          "history": inputs,
          "n" : 1,
          "best_of": 1, 
          "presence_penalty": 1.2, 
          "frequency_penalty": 0.2, 
          "temperature": 0.3, 
          "top_p" : 0.95, 
          "top_k": 50, 
          "use_beam_search": False, 
          "stop": [], 
          "ignore_eos" :False, 
          "logprobs": None,
          "max_new_tokens": 2048, 
    }
    request = urllib.request.Request(
        url='http://127.0.0.1:8001/generate',
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
    inputs = [ {"role" : "Human", "content": last_question}]
    
    test_api_server(inputs)

