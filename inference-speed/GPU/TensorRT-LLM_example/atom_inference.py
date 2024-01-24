import csv
import sys
from pathlib import Path

import numpy as np
import torch
from utils import (DEFAULT_HF_MODEL_DIRS, DEFAULT_PROMPT_TEMPLATES,
                   load_tokenizer, read_model_name, throttle_generator)

import tensorrt_llm
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime import PYTHON_BINDINGS, ModelRunner

if PYTHON_BINDINGS:
    from tensorrt_llm.runtime import ModelRunnerCpp

class AtomTRTApi:
    def __init__(self,engine_dir,tokenizer_dir,max_input_length=4096):
        self.runtime_rank = tensorrt_llm.mpi_rank()
        self.model_name = read_model_name(engine_dir)

        self.tokenizer, self.pad_id, self.end_id = load_tokenizer(
            tokenizer_dir=tokenizer_dir,
            tokenizer_type='llama',
        )
        self.use_py_session=False
        if not PYTHON_BINDINGS:
            logger.warning(
                "Python bindings of C++ session is unavailable, fallback to Python session."
            )
            self.use_py_session = True
        runner_cls = ModelRunner if self.use_py_session else ModelRunnerCpp
        runner_kwargs = dict(engine_dir=engine_dir,
                            lora_dir=None,
                            rank=self.runtime_rank,
                            debug_mode=False,
                            lora_ckpt_source='hf')
        
        if not self.use_py_session:
            runner_kwargs.update(
                max_batch_size=1,
                max_input_len=max_input_length,
                max_output_len=2048,
                max_beam_width=1,
                max_attention_window_size=None)
        self.runner = runner_cls.from_dir(**runner_kwargs)


    def ask(self,input_text,temperature=0.4,top_p=0.95,max_new_tokens=1024,repetition_penalty=1.2,system_prefix = '',merge_lambda=None,max_input_length=4096,append_next_role=True):
        with torch.no_grad():
            prompt = ''
            print('max_input_length',max_input_length)
            if type(input_text)==list:
                for input_text_one in input_text[::-1]:
                    if len(prompt) + len("<s>"+input_text_one['role']+": "+input_text_one['content'].strip()+"\n</s>")<max_input_length:
                        prompt = "<s>"+input_text_one['role']+": "+input_text_one['content'].strip()+"\n</s>" + prompt
                if append_next_role:
                    if input_text[-1]['role']=='Human':
                        prompt += "<s>Assistant:"
                    else:
                        prompt += "<s>Human:"
            else:
                if merge_lambda is None:
                    if append_next_role:
                        prompt +=  "<s>Human: "+input_text.strip()+"\n</s><s>Assistant:"
                    else:
                        prompt +=  "<s>Human: "+input_text.strip()+"\n</s>"
                else:
                    prompt +=  merge_lambda(input_text)
            if len(system_prefix)>0:
                prompt = '<s>System: '+system_prefix.strip()+'\n</s>'+prompt
            print('输入模型的完整输入:',prompt)
            input_ids = [self.tokenizer(prompt,add_special_tokens=False).input_ids]
            print(input_ids)
            input_ids = [
                torch.tensor(x, dtype=torch.int32).unsqueeze(0) for x in input_ids
            ]
            print('输入模型的token数量',input_ids[0].shape)
            generate_input = {
                "batch_input_ids":input_ids,
                "max_new_tokens":max_new_tokens,
                "max_attention_window_size":None,
                "do_sample":True,
                "top_k":50,
                "top_p":top_p,
                "num_beams":1,
                "length_penalty":1.0,
                "stop_words_list":None,
                "bad_words_list":None,
                "streaming":False,
                "temperature":temperature,
                "output_sequence_lengths":True,
                "return_dict":False,
                "repetition_penalty":repetition_penalty,
                "end_id":self.tokenizer.eos_token_id,
                "bos_token_id":self.tokenizer.bos_token_id,
                "pad_id":self.tokenizer.pad_token_id
            }
            generate_ids = self.runner.generate(**generate_input)
            torch.cuda.synchronize()
            print(generate_ids)
            generate_ids = generate_ids.cpu().tolist()
            generate_ids = [item[0][len(input_ids[0][0]):] for  item in generate_ids]
            try:
                generate_ids = [item[:item.index(self.tokenizer.eos_token_id)] for  item in generate_ids ]
            except:
                pass
            print(generate_ids)
            # output = ''.join(tokenizer.convert_ids_to_tokens(generate_ids[0]))
            # print('生成的token长度',len(generate_ids[0]))
            bot_message = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
            if 'Human:' in bot_message:
                bot_message = bot_message.split('Human:')[0]
            print(bot_message)
            return bot_message.strip()
        
    def ask_streaming(self,input_text,temperature=0.8,top_p=0.95,max_new_tokens=1024,repetition_penalty=1.2,system_prefix = '',max_input_length=4096,append_next_role=True):
        with torch.no_grad():
            prompt = ''
            print('max_input_length',max_input_length)
            if type(input_text)==list:
                for input_text_one in input_text[::-1]:
                    if len(prompt) + len("<s>"+input_text_one['role']+": "+input_text_one['content'].strip()+"\n</s>")<max_input_length:
                        prompt = "<s>"+input_text_one['role']+": "+input_text_one['content'].strip()+"\n</s>" + prompt
                if append_next_role:
                    if input_text[-1]['role']=='Human':
                        prompt += "<s>Assistant:"
                    else:
                        prompt += "<s>Human:"
            else:
                if append_next_role:
                    prompt +=  "<s>Human: "+input_text.strip()+"\n</s><s>Assistant:"
                else:
                    prompt +=  "<s>Human: "+input_text.strip()+"\n</s>"
            if len(system_prefix)>0:
                prompt = '<s>System: '+system_prefix.strip()+'\n</s>'+prompt
            print('输入模型的完整输入:',prompt)
            input_ids = [self.tokenizer(prompt,add_special_tokens=False).input_ids]
            print(input_ids)
            input_ids = [
                torch.tensor(x, dtype=torch.int32).unsqueeze(0) for x in input_ids
            ]
            print('输入模型的token数量',input_ids[0].shape)
            generate_input = {
                "batch_input_ids":input_ids,
                "max_new_tokens":max_new_tokens,
                "max_attention_window_size":None,
                "do_sample":True,
                "top_k":50,
                "top_p":top_p,
                "num_beams":1,
                "length_penalty":1.0,
                "stop_words_list":None,
                "bad_words_list":None,
                "streaming":True,
                "temperature":temperature,
                "output_sequence_lengths":True,
                "return_dict":True,
                "repetition_penalty":repetition_penalty,
                "end_id":self.tokenizer.eos_token_id,
                "bos_token_id":self.tokenizer.bos_token_id,
                "pad_id":self.tokenizer.pad_token_id
            }
            generate_ids = self.runner.generate(**generate_input)
            torch.cuda.synchronize()
            
            input_token_num = len(input_ids[0][0])
            answer_message =''
            for curr_outputs in throttle_generator(generate_ids,2):
                output_ids = curr_outputs['output_ids']
                sequence_lengths = curr_outputs['sequence_lengths']
                # print(sequence_lengths)
                output_ids = output_ids.cpu().tolist()
                output_ids = [item[0][input_token_num:sequence_lengths[0][0]] for  item in output_ids]
                answer_message = self.tokenizer.batch_decode(output_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
                if 'Human:' in answer_message:
                    answer_message = answer_message.split('Human:')[0]                
                yield answer_message.strip()
            return answer_message.strip()
            
            
if __name__=='__main__':
    model = AtomTRTApi(engine_dir=sys.argv[1],tokenizer_dir=sys.argv[2])
    model.ask('如何成为一个更优秀的人')
