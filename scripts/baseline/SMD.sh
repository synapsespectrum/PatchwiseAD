#!/bin/bash

# Execute the python script with arguments directly, using updated values
python run.py \
     --is_pretrain 0 \
    --is_finetune 1 \
    --is_training 1 \
    --model AnomalyBERT \
    --dataset SMD \
    --window_sliding 16 \
    --soft_replacing 0.5 \
    --flip_replacing_interval all \
    --uniform_replacing 0.15 \
    --peak_noising 0.15 \
    --length_adjusting 0.1 \
    --white_noising 0.0 \
    --batch_size 16 \
    --learning_rate 0.0001 \
    --grad_clip_norm 1.0 \
    --replacing_rate_max 0.2 \
    --replacing_weight 0.7 \
    --input_encoder_len 512 \
    --patch_size 4 \
    --e_layers 6 \
    --train_epochs 5