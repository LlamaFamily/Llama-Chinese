CUDA_VISIBLE_DEVICES=0,3 python api_server.py \
--model "/mnt/data/zhangzheng/data/atomgpt/model_sft_atomgpt_28000_0613/checkpoint-12000_merge" \
--port 8090 \
--tensor-parallel-size 2
