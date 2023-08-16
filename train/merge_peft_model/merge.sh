CUDA_VISIBLE_DEVICES=0 python merge_peft_adapter.py \
    --adapter_model_name /checkpoint-2200 \
    --output_name checkpoint-2200_merge \
    --load8bit false \
    --tokenizer_fast false  