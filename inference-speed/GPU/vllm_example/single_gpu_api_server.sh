# /mnt/data/zhangzheng/data/atomgpt/model_sft_atomgpt_28000_0613/checkpoint-12000_merge

CUDA_VISIBLE_DEVICES=0 python api_server.py \
--model "/mnt/data_online/models/llama/models--meta-llama--Llama-2-7b-chat-hf" \
--port 8090
