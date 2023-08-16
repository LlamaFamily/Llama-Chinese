from dataclasses import dataclass, field
from typing import Optional,List

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

    adapter_model_name: Optional[List[str]] = field(default=None, metadata={"help": "the model name"})
    output_name: Optional[str] = field(default=None, metadata={"help": "the model name"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

base_model = None
for one_lora_path in script_args.adapter_model_name:
    if base_model==None:
        peft_config = PeftConfig.from_pretrained(one_lora_path)
        tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
        tokenizer.save_pretrained(f"{script_args.output_name}")
        base_model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path, return_dict=True, torch_dtype=torch.bfloat16)
    peft_config = PeftConfig.from_pretrained(one_lora_path)
    base_model = PeftModel.from_pretrained(base_model, one_lora_path,device_map={"": 0})
    # model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path, return_dict=True, device_map='auto',load_in_8bit=True)
    # Load the Lora model
    base_model = base_model.merge_and_unload()
    base_model.eval()




# key_list = [key for key, _ in model.base_model.model.named_modules() if "lora" not in key]
# for key in key_list:
#     print(key)
#     parent, target, target_name = _get_submodules(model.base_model,key)
#     if isinstance(target, peft.tuners.lora.Linear):
#         print('peft.tuners.lora.Linear')
#         bias = target.bias is not None
#         new_module = torch.nn.Linear(target.in_features, target.out_features, bias=bias)
#         model.base_model._replace_module(parent, target_name, new_module, target)

# model = model.base_model.model


base_model.save_pretrained(f"{script_args.output_name}")
# model.push_to_hub(f"{script_args.output_name}", use_temp_dir=False)