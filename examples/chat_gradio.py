import gradio as gr
import time
from transformers import AutoTokenizer, AutoModelForCausalLM,TextIteratorStreamer
from threading import Thread
import torch,sys,os
import json
import pandas 
import argparse

with gr.Blocks() as demo:
    gr.Markdown("""<h1><center>智能助手</center></h1>""")
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    state = gr.State()
    with gr.Row():
        clear = gr.Button("新话题")
        re_generate = gr.Button("重新回答")
        sent_bt = gr.Button("发送")
    with gr.Accordion("生成参数", open=False):
        slider_temp = gr.Slider(minimum=0, maximum=1, label="temperature", value=0.3)
        slider_top_p = gr.Slider(minimum=0.5, maximum=1, label="top_p", value=0.95)
        slider_context_times = gr.Slider(minimum=0, maximum=5, label="上文轮次", value=0,step=2.0)
    def user(user_message, history):
        return "", history + [[user_message, None]]
    def bot(history,temperature,top_p,slider_context_times):
        if pandas.isnull(history[-1][1])==False:
            history[-1][1] = None
            yield history
        slider_context_times = int(slider_context_times)
        history_true = history[1:-1]
        prompt = ''
        if slider_context_times>0:
            prompt += '\n'.join([("<s>Human: "+one_chat[0].replace('<br>','\n')+'\n</s>' if one_chat[0] else '')  +"<s>Assistant: "+one_chat[1].replace('<br>','\n')+'\n</s>'    for one_chat in history_true[-slider_context_times:] ])
        prompt +=  "<s>Human: "+history[-1][0].replace('<br>','\n')+"\n</s><s>Assistant:"
        input_ids = tokenizer([prompt], return_tensors="pt",add_special_tokens=False).input_ids[:,-512:].to('cuda')        
        generate_input = {
            "input_ids":input_ids,
            "max_new_tokens":512,
            "do_sample":True,
            "top_k":50,
            "top_p":top_p,
            "temperature":temperature,
            "repetition_penalty":1.3,
            "streamer":streamer,
            "eos_token_id":tokenizer.eos_token_id,
            "bos_token_id":tokenizer.bos_token_id,
            "pad_token_id":tokenizer.pad_token_id
        }
        thread = Thread(target=model.generate, kwargs=generate_input)
        thread.start()
        start_time = time.time()
        bot_message =''
        print('Human:',history[-1][0])
        print('Assistant: ',end='',flush=True)
        for new_text in streamer:
            print(new_text,end='',flush=True)
            if len(new_text)==0:
                continue
            if new_text!='</s>':
                bot_message+=new_text
            if 'Human:' in bot_message:
                bot_message = bot_message.split('Human:')[0]
            history[-1][1] = bot_message
            yield history
        end_time =time.time()
        print()
        print('生成耗时：',end_time-start_time,'文字长度：',len(bot_message),'字耗时：',(end_time-start_time)/len(bot_message))

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, [chatbot,slider_temp,slider_top_p,slider_context_times], chatbot
    )
    sent_bt.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, [chatbot,slider_temp,slider_top_p,slider_context_times], chatbot
    )
    re_generate.click( bot, [chatbot,slider_temp,slider_top_p,slider_context_times], chatbot )
    clear.click(lambda: [], None, chatbot, queue=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, help='mode name or path')
    parser.add_argument("--is_4bit", action='store_true', help='use 4bit model')
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    if args.is_4bit==False:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,device_map='auto',torch_dtype=torch.float16,load_in_8bit=True)
        model.eval()
    else:
        from auto_gptq import AutoGPTQForCausalLM
        model = AutoGPTQForCausalLM.from_quantized(args.model_name_or_path,low_cpu_mem_usage=True, device="cuda:0", use_triton=False,inject_fused_attention=False,inject_fused_mlp=False)
    streamer = TextIteratorStreamer(tokenizer,skip_prompt=True)
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    demo.queue().launch(share=False, debug=True,server_name="0.0.0.0")
