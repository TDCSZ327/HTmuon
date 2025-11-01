#!/bin/bash
# bash hed_scripts/60M/adam/adam_60M_lr1e_3_WD1e_5.sh


export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
export CUDA_VISIBLE_DEVICES=2,3
torchrun --nproc_per_node=2 --master_port=20018 --master_addr=localhost torchrun_main_muon.py \
    --model_config configs/llama_60m.json \
    --optimizer muon_orth \
    --seed 5 \
    --lr 0.001 \
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps 20000 \
    --warmup_steps 2000 \
    --weight_decay 0.1 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --wandb_name muon-60M-lr0.001-WD0.1-repeat1 \
    --target_eval_tokens 10_000_000 \
    --save_every 10000

export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
export CUDA_VISIBLE_DEVICES=2,3
torchrun --nproc_per_node=2 --master_port=20019 --master_addr=localhost torchrun_main_muon.py \
    --model_config configs/llama_60m.json \
    --optimizer muon_orth \
    --seed 6 \
    --lr 0.001 \
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps 20000 \
    --warmup_steps 2000 \
    --weight_decay 0.1 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --wandb_name muon-60M-lr0.001-WD0.1-repeat2 \
    --target_eval_tokens 10_000_000 \
    --save_every 10000

export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
export CUDA_VISIBLE_DEVICES=2,3
torchrun --nproc_per_node=2 --master_port=20020 --master_addr=localhost torchrun_main_muon.py \
    --model_config configs/llama_60m.json \
    --optimizer muon_orth \
    --seed 7 \
    --lr 0.001 \
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps 20000 \
    --warmup_steps 2000 \
    --weight_decay 0.1 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --wandb_name muon-60M-lr0.001-WD0.1-repeat3 \
    --target_eval_tokens 10_000_000 \
    --save_every 10000

