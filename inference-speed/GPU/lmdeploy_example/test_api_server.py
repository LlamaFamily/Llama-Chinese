import os
import time
from lmdeploy.serve.turbomind.chatbot import Chatbot

def input_prompt(chat_history, system_prompt: str):
    """Input a prompt in the consolo interface."""
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

def main(tritonserver_addr: str,
         session_id: int = 1,
         cap: str = 'chat',
         stream_output: bool = True,
         **kwargs):
    """An example to communicate with inference server through the command line
    interface.

    Args:
        tritonserver_addr (str): the address in format "ip:port" of
          triton inference server
        session_id (int): the identical id of a session
        cap (str): the capability of a model. For example, codellama has
            the ability among ['completion', 'infill', 'instruct', 'python']
        stream_output (bool): indicator for streaming output or not
        **kwargs (dict): other arguments for initializing model's chat template
    """
    log_level = os.environ.get('SERVICE_LOG_LEVEL', 'WARNING')
    kwargs.update(capability=cap)
    chatbot = Chatbot(tritonserver_addr,
                      log_level=log_level,
                      display=stream_output,
                      **kwargs)
    nth_round = 1
    prompt = input_prompt([{"role": "Human", "content" : "心情不好怎么办"}], "")

    request_id = f'{session_id}-{nth_round}'
    begin = time.time()
    if stream_output:
        for status, res, n_token in chatbot.stream_infer(
                session_id,
                prompt,
                request_id=request_id,
                request_output_len=512):
            # print("n_token:", n_token)
            continue
            
    else:
        status, res, n_token = chatbot.infer(session_id,
                                                prompt,
                                                request_id=request_id,
                                                request_output_len=512)
        print(res)
        # print("n_token:", n_token)
    nth_round += 1
    end = time.time()
    speed = n_token/(end-begin)
    print("speed {} tokens/s".format(speed))
    

if __name__ == '__main__':
    import fire

    fire.Fire(main)
