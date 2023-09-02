## Meta官网模型权重转换成Hugging Face格式

使用脚本
```bash
python convert_llama_weights_to_hf.py \
    --input_dir /path/to/downloaded/llama/weights \
    --model_size 7B \
    --output_dir /output/path
```

通过脚本转换后的模型权重可以使用transformers进行加载，例如：

```py
from transformers import LlamaForCausalLM, LlamaTokenizer

model = LlamaForCausalLM.from_pretrained("/output/path")
tokenizer = LlamaTokenizer.from_pretrained("/output/path")
```