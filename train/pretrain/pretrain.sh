output_model=/mnt/data1/atomgpt
if [ ! -d ${output_model} ];then  
    mkdir ${output_model}
fi
cp ./pretrain.sh ${output_model}
cp ./ds_config_zero*.json ${output_model}

deepspeed --num_gpus 8 pretrain_clm.py \
    --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
    --train_files ../../data/train_sft.csv \
                ../../data/train_sft_sharegpt.csv \
    --validation_files  ../../data/dev_sft.csv \
                         ../../data/dev_sft_sharegpt.csv \
    --per_device_train_batch_size 10 \
    --per_device_eval_batch_size 10 \
    --do_train \
    --output_dir ${output_model} \
    --evaluation_strategy  steps \
    --use_fast_tokenizer false \
    --max_eval_samples 500 \
    --learning_rate 3e-5 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 3 \
    --warmup_steps 10000 \
    --logging_dir ${output_model}/logs \
    --logging_strategy steps \
    --logging_steps 2 \
    --save_strategy steps \
    --preprocessing_num_workers 10 \
    --save_steps 500 \
    --eval_steps 500 \
    --save_total_limit 2000 \
    --seed 42 \
    --disable_tqdm false \
    --ddp_find_unused_parameters false \
    --block_size 4096 \
    --overwrite_output_dir \
    --report_to tensorboard \
    --run_name ${output_model} \
    --bf16 \
    --bf16_full_eval \
    --gradient_checkpointing \
    --deepspeed ./ds_config_zero3.json \
    --ignore_data_skip true \
    --ddp_timeout 18000000 \
    | tee -a ${output_model}/train.log
    
    # --resume_from_checkpoint ${output_model}/checkpoint-20400 \
