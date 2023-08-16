CUDA_VISIBLE_DEVICES=3 python merge_muilt_peft_adapter.py \
    --adapter_model_name checkpoint-8000 \
                    checkpoint-140 \
    --output_name checkpoint-140-8000_merge 