#!/bin/bash
# bash hed_scripts/60M/adam/adam_60M_lr1e_3_WD1e_5.sh


export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 --master_port=20001 --master_addr=localhost torchrun_main.py \
    --model_config configs/llama_60m.json \
    --optimizer adam \
    --seed 5 \
    --lr 0.001 \
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps 20000 \
    --warmup_steps 2000 \
    --weight_decay 0.00001 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --wandb_name adam-60M-lr0.001-WD0.00001-repeat1 \
    --target_eval_tokens 10_000_000 \
    --save_every 10000

export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 --master_port=20002 --master_addr=localhost torchrun_main.py \
    --model_config configs/llama_60m.json \
    --optimizer adam \
    --seed 6 \
    --lr 0.001 \
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps 20000 \
    --warmup_steps 2000 \
    --weight_decay 0.00001 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --wandb_name adam-60M-lr0.001-WD0.00001-repeat2 \
    --target_eval_tokens 10_000_000 \
    --save_every 10000

export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 --master_port=20003 --master_addr=localhost torchrun_main.py \
    --model_config configs/llama_60m.json \
    --optimizer adam \
    --seed 7 \
    --lr 0.001 \
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps 20000 \
    --warmup_steps 2000 \
    --weight_decay 0.00001 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --wandb_name adam-60M-lr0.001-WD0.00001-repeat3 \
    --target_eval_tokens 10_000_000 \
    --save_every 10000

