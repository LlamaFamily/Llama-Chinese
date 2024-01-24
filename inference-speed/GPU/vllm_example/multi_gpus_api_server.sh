CUDA_VISIBLE_DEVICES=0,1 python api_server.py \
--model "./Atom-7B-Chat" \
--port 8090 \
--tensor-parallel-size 2
