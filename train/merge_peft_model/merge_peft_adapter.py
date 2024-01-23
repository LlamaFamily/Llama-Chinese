from dataclasses import dataclass, field
from typing import Optional

import peft
import torch
from peft import PeftConfig, PeftModel,PeftModelForSequenceClassification
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, HfArgumentParser,AutoModelForSequenceClassification
from peft.utils import _get_submodules

@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    adapter_model_name: Optional[str] = field(default=None, metadata={"help": "the model name"})
    load8bit : Optional[bool] = field(default=None, metadata={"help": "the model type"})
    output_name: Optional[str] = field(default=None, metadata={"help": "the model name"})
    tokenizer_fast:Optional[bool] = field(default=None, metadata={"help": "the model type"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]


peft_config = PeftConfig.from_pretrained(script_args.adapter_model_name)
model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path, return_dict=True, torch_dtype=torch.float16,device_map='auto',trust_remote_code=True)
model = PeftModel.from_pretrained(model, script_args.adapter_model_name,device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path,use_fast=script_args.tokenizer_fast)
config = AutoConfig.from_pretrained(peft_config.base_model_name_or_path)
architecture = config.architectures[0]
print(architecture)
# Load the Lora model
model = model.merge_and_unload()
model.eval()


model.save_pretrained(f"{script_args.output_name}")
tokenizer.save_pretrained(f"{script_args.output_name}")
if script_args.load8bit:
    model = AutoModelForCausalLM.from_pretrained(script_args.output_name, torch_dtype=torch.float16,load_in_8bit=script_args.load8bit,device_map='auto',trust_remote_code=True)
    model.save_pretrained(f"{script_args.output_name}",max_shard_size='5GB')