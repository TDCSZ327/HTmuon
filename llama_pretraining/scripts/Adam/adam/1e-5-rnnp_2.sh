#!/bin/bash
# bash hed_scripts/60M/adam/adam_60M_lr1e_3_WD1e_5.sh


# export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
# export CUDA_VISIBLE_DEVICES=0,3
# torchrun --nproc_per_node=2 --master_port=21779 --master_addr=localhost torchrun_main_muon_test.py \
#     --model_config configs/llama_60m.json \
#     --optimizer rnnp \
#     --seed 5 \
#     --lr 0.001 \
#     --lrmuon 0.005\
#     --power 0.25 \
#     --batch_size 128 \
#     --total_batch_size 512 \
#     --num_training_steps 20000 \
#     --warmup_steps 2000 \
#     --weight_decay 0.1 \
#     --dtype bfloat16 \
#     --eval_every 1000 \
#     --wandb_name rnnp-60M-lr0.001-lrmuon0.005-WD0.1-bz128-step20000-repeat1 \
#     --target_eval_tokens 10_000_000 \
#     --save_every 10000


# export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
# export CUDA_VISIBLE_DEVICES=0,3
# torchrun --nproc_per_node=2 --master_port=21456 --master_addr=localhost torchrun_main_muon_test.py \
#     --model_config configs/llama_60m.json \
#     --optimizer rnnp \
#     --seed 5 \
#     --lr 0.001 \
#     --lrmuon 0.005\
#     --power 0.25 \
#     --batch_size 128 \
#     --total_batch_size 512 \
#     --num_training_steps 10000 \
#     --warmup_steps 1000 \
#     --weight_decay 0.1 \
#     --dtype bfloat16 \
#     --eval_every 1000 \
#     --wandb_name rnnp-60M-lr0.001-lrmuon0.005-WD0.1-bz128-step10000-repeat1 \
#     --target_eval_tokens 10_000_000 \
#     --save_every 10000


# export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
# export CUDA_VISIBLE_DEVICES=0,3
# torchrun --nproc_per_node=2 --master_port=21178 --master_addr=localhost torchrun_main_muon_test.py \
#     --model_config configs/llama_60m.json \
#     --optimizer muon \
#     --seed 5 \
#     --lr 0.001 \
#     --lrmuon 0.01\
#     --power 0.25 \
#     --batch_size 128 \
#     --total_batch_size 512 \
#     --num_training_steps 10000 \
#     --warmup_steps 1000 \
#     --weight_decay 0.1 \
#     --dtype bfloat16 \
#     --eval_every 1000 \
#     --wandb_name muon-60M-lr0.001-lrmuon0.01-WD0.1-bz-128-step10000-repeat1 \
#     --target_eval_tokens 10_000_000 \
#     --save_every 10000


# export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
# export CUDA_VISIBLE_DEVICES=0,3
# torchrun --nproc_per_node=2 --master_port=22107 --master_addr=localhost torchrun_main_muon_test.py \
#     --model_config configs/llama_60m.json \
#     --optimizer muon_generalized \
#     --seed 5 \
#     --lr 0.001 \
#     --lrmuon 0.01\
#     --power 0.25 \
#     --batch_size 128 \
#     --total_batch_size 512 \
#     --num_training_steps 10000 \
#     --warmup_steps 1000 \
#     --weight_decay 0.1 \
#     --dtype bfloat16 \
#     --eval_every 1000 \
#     --wandb_name muonG-60M-lr0.001-lrmuon0.01-WD0.1-bz128-step10000-repeat1 \
#     --target_eval_tokens 10_000_000 \
#     --save_every 10000


# export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
# export CUDA_VISIBLE_DEVICES=0,3
# torchrun --nproc_per_node=2 --master_port=21178 --master_addr=localhost torchrun_main_muon_test.py \
#     --model_config configs/llama_60m.json \
#     --optimizer muon \
#     --seed 5 \
#     --lr 0.001 \
#     --lrmuon 0.02\
#     --power 0.25 \
#     --batch_size 128 \
#     --total_batch_size 512 \
#     --num_training_steps 10000 \
#     --warmup_steps 1000 \
#     --weight_decay 0.1 \
#     --dtype bfloat16 \
#     --eval_every 1000 \
#     --wandb_name muon-60M-lr0.001-lrmuon0.02-WD0.1-bz-128-step10000-repeat1 \
#     --target_eval_tokens 10_000_000 \
#     --save_every 10000


# export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
# export CUDA_VISIBLE_DEVICES=0,3
# torchrun --nproc_per_node=2 --master_port=22107 --master_addr=localhost torchrun_main_muon_test.py \
#     --model_config configs/llama_60m.json \
#     --optimizer muon_generalized \
#     --seed 5 \
#     --lr 0.001 \
#     --lrmuon 0.02\
#     --power 0.25 \
#     --batch_size 128 \
#     --total_batch_size 512 \
#     --num_training_steps 10000 \
#     --warmup_steps 1000 \
#     --weight_decay 0.1 \
#     --dtype bfloat16 \
#     --eval_every 1000 \
#     --wandb_name muonG-60M-lr0.001-lrmuon0.02-WD0.1-bz128-step10000-repeat1 \
#     --target_eval_tokens 10_000_000 \
#     --save_every 10000



# export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
# export CUDA_VISIBLE_DEVICES=2,3
# torchrun --nproc_per_node=2 --master_port=21178 --master_addr=localhost torchrun_main_muon_test.py \
#     --model_config configs/llama_60m.json \
#     --optimizer muon \
#     --seed 5 \
#     --lr 0.001 \
#     --lrmuon 0.03\
#     --power 0.25 \
#     --batch_size 256 \
#     --total_batch_size 512 \
#     --num_training_steps 20000 \
#     --warmup_steps 2000 \
#     --weight_decay 0.1 \
#     --dtype bfloat16 \
#     --eval_every 1000 \
#     --wandb_name muon-60M-lr0.001-lrmuon0.03-WD0.1-repeat1 \
#     --target_eval_tokens 10_000_000 \
#     --save_every 10000


# export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
# export CUDA_VISIBLE_DEVICES=2,3
# torchrun --nproc_per_node=2 --master_port=22107 --master_addr=localhost torchrun_main_muon_test.py \
#     --model_config configs/llama_60m.json \
#     --optimizer muon_generalized \
#     --seed 5 \
#     --lr 0.001 \
#     --lrmuon 0.03\
#     --power 0.25 \
#     --batch_size 256 \
#     --total_batch_size 512 \
#     --num_training_steps 20000 \
#     --warmup_steps 2000 \
#     --weight_decay 0.1 \
#     --dtype bfloat16 \
#     --eval_every 1000 \
#     --wandb_name muonG-60M-lr0.001-lrmuon0.03-WD0.1-repeat1 \
#     --target_eval_tokens 10_000_000 \
#     --save_every 10000




# export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
# export CUDA_VISIBLE_DEVICES=0,3
# torchrun --nproc_per_node=2 --master_port=21178 --master_addr=localhost torchrun_main_muon_test.py \
#     --model_config configs/llama_60m.json \
#     --optimizer muon \
#     --seed 5 \
#     --lr 0.001 \
#     --lrmuon 0.04\
#     --power 0.25 \
#     --batch_size 256 \
#     --total_batch_size 512 \
#     --num_training_steps 20000 \
#     --warmup_steps 2000 \
#     --weight_decay 0.1 \
#     --dtype bfloat16 \
#     --eval_every 1000 \
#     --wandb_name muon-60M-lr0.001-lrmuon0.04-WD0.1-repeat1 \
#     --target_eval_tokens 10_000_000 \
#     --save_every 10000


# export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
# export CUDA_VISIBLE_DEVICES=0,3
# torchrun --nproc_per_node=2 --master_port=22107 --master_addr=localhost torchrun_main_muon_test.py \
#     --model_config configs/llama_60m.json \
#     --optimizer muon_generalized \
#     --seed 5 \
#     --lr 0.001 \
#     --lrmuon 0.04\
#     --power 0.25 \
#     --batch_size 256 \
#     --total_batch_size 512 \
#     --num_training_steps 20000 \
#     --warmup_steps 2000 \
#     --weight_decay 0.1 \
#     --dtype bfloat16 \
#     --eval_every 1000 \
#     --wandb_name muonG-60M-lr0.001-lrmuon0.04-WD0.1-repeat1 \
#     --target_eval_tokens 10_000_000 \
#     --save_every 10000







# export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
# export CUDA_VISIBLE_DEVICES=0,3
# torchrun --nproc_per_node=2 --master_port=21178 --master_addr=localhost torchrun_main_muon_test.py \
#     --model_config configs/llama_60m.json \
#     --optimizer muon \
#     --seed 5 \
#     --lr 0.001 \
#     --lrmuon 0.01\
#     --power 0.25 \
#     --batch_size 256 \
#     --total_batch_size 512 \
#     --num_training_steps 20000 \
#     --warmup_steps 2000 \
#     --weight_decay 0.1 \
#     --dtype bfloat16 \
#     --eval_every 1000 \
#     --wandb_name muon-60M-lr0.001-lrmuon0.01-WD0.1-repeat1 \
#     --target_eval_tokens 10_000_000 \
#     --save_every 10000


# export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
# export CUDA_VISIBLE_DEVICES=0,3
# torchrun --nproc_per_node=2 --master_port=22107 --master_addr=localhost torchrun_main_muon_test.py \
#     --model_config configs/llama_60m.json \
#     --optimizer muon_generalized \
#     --seed 5 \
#     --lr 0.001 \
#     --lrmuon 0.01\
#     --power 0.25 \
#     --batch_size 256 \
#     --total_batch_size 512 \
#     --num_training_steps 20000 \
#     --warmup_steps 2000 \
#     --weight_decay 0.1 \
#     --dtype bfloat16 \
#     --eval_every 1000 \
#     --wandb_name muonG-60M-lr0.001-lrmuon0.01-WD0.1-repeat1 \
#     --target_eval_tokens 10_000_000 \
#     --save_every 10000




# export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
# export CUDA_VISIBLE_DEVICES=0,3
# torchrun --nproc_per_node=2 --master_port=21178 --master_addr=localhost torchrun_main_muon_test.py \
#     --model_config configs/llama_60m.json \
#     --optimizer muon \
#     --seed 5 \
#     --lr 0.001 \
#     --lrmuon 0.02\
#     --power 0.25 \
#     --batch_size 256 \
#     --total_batch_size 512 \
#     --num_training_steps 10000 \
#     --warmup_steps 1000 \
#     --weight_decay 0.1 \
#     --dtype bfloat16 \
#     --eval_every 1000 \
#     --wandb_name muon-60M-lr0.001-lrmuon0.02-WD0.1-step10000-repeat1 \
#     --target_eval_tokens 10_000_000 \
#     --save_every 10000


# export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
# export CUDA_VISIBLE_DEVICES=0,3
# torchrun --nproc_per_node=2 --master_port=22107 --master_addr=localhost torchrun_main_muon_test.py \
#     --model_config configs/llama_60m.json \
#     --optimizer muon_generalized \
#     --seed 5 \
#     --lr 0.001 \
#     --lrmuon 0.02\
#     --power 0.25 \
#     --batch_size 256 \
#     --total_batch_size 512 \
#     --num_training_steps 10000 \
#     --warmup_steps 1000 \
#     --weight_decay 0.1 \
#     --dtype bfloat16 \
#     --eval_every 1000 \
#     --wandb_name muonG-60M-lr0.001-lrmuon0.02-WD0.1-step10000-repeat1 \
#     --target_eval_tokens 10_000_000 \
#     --save_every 10000







# export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
# export CUDA_VISIBLE_DEVICES=0,3
# torchrun --nproc_per_node=2 --master_port=21178 --master_addr=localhost torchrun_main_muon_test.py \
#     --model_config configs/llama_60m.json \
#     --optimizer muon \
#     --seed 5 \
#     --lr 0.001 \
#     --lrmuon 0.03\
#     --power 0.25 \
#     --batch_size 256 \
#     --total_batch_size 512 \
#     --num_training_steps 10000 \
#     --warmup_steps 1000 \
#     --weight_decay 0.1 \
#     --dtype bfloat16 \
#     --eval_every 1000 \
#     --wandb_name muon-60M-lr0.001-lrmuon0.03-WD0.1-step10000-repeat1 \
#     --target_eval_tokens 10_000_000 \
#     --save_every 10000


# export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
# export CUDA_VISIBLE_DEVICES=0,3
# torchrun --nproc_per_node=2 --master_port=22107 --master_addr=localhost torchrun_main_muon_test.py \
#     --model_config configs/llama_60m.json \
#     --optimizer muon_generalized \
#     --seed 5 \
#     --lr 0.001 \
#     --lrmuon 0.03\
#     --power 0.25 \
#     --batch_size 256 \
#     --total_batch_size 512 \
#     --num_training_steps 10000 \
#     --warmup_steps 1000 \
#     --weight_decay 0.1 \
#     --dtype bfloat16 \
#     --eval_every 1000 \
#     --wandb_name muonG-60M-lr0.001-lrmuon0.03-WD0.1-step10000-repeat1 \
#     --target_eval_tokens 10_000_000 \
#     --save_every 10000







# export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
# export CUDA_VISIBLE_DEVICES=0,3
# torchrun --nproc_per_node=2 --master_port=21178 --master_addr=localhost torchrun_main_muon_test.py \
#     --model_config configs/llama_60m.json \
#     --optimizer muon \
#     --seed 5 \
#     --lr 0.001 \
#     --lrmuon 0.04\
#     --power 0.25 \
#     --batch_size 256 \
#     --total_batch_size 512 \
#     --num_training_steps 10000 \
#     --warmup_steps 1000 \
#     --weight_decay 0.1 \
#     --dtype bfloat16 \
#     --eval_every 1000 \
#     --wandb_name muon-60M-lr0.001-lrmuon0.04-WD0.1-step10000-repeat1 \
#     --target_eval_tokens 10_000_000 \
#     --save_every 10000


# export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
# export CUDA_VISIBLE_DEVICES=0,3
# torchrun --nproc_per_node=2 --master_port=22107 --master_addr=localhost torchrun_main_muon_test.py \
#     --model_config configs/llama_60m.json \
#     --optimizer muon_generalized \
#     --seed 5 \
#     --lr 0.001 \
#     --lrmuon 0.04\
#     --power 0.25 \
#     --batch_size 256 \
#     --total_batch_size 512 \
#     --num_training_steps 10000 \
#     --warmup_steps 1000 \
#     --weight_decay 0.1 \
#     --dtype bfloat16 \
#     --eval_every 1000 \
#     --wandb_name muonG-60M-lr0.001-lrmuon0.04-WD0.1-step10000-repeat1 \
#     --target_eval_tokens 10_000_000 \
#     --save_every 10000







# export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
# export CUDA_VISIBLE_DEVICES=0,3
# torchrun --nproc_per_node=2 --master_port=21178 --master_addr=localhost torchrun_main_muon_test.py \
#     --model_config configs/llama_60m.json \
#     --optimizer muon \
#     --seed 5 \
#     --lr 0.001 \
#     --lrmuon 0.01\
#     --power 0.25 \
#     --batch_size 256 \
#     --total_batch_size 512 \
#     --num_training_steps 10000 \
#     --warmup_steps 1000 \
#     --weight_decay 0.1 \
#     --dtype bfloat16 \
#     --eval_every 1000 \
#     --wandb_name muon-60M-lr0.001-lrmuon0.01-WD0.1-step10000-repeat1 \
#     --target_eval_tokens 10_000_000 \
#     --save_every 10000


# export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
# export CUDA_VISIBLE_DEVICES=0,3
# torchrun --nproc_per_node=2 --master_port=22107 --master_addr=localhost torchrun_main_muon_test.py \
#     --model_config configs/llama_60m.json \
#     --optimizer muon_generalized \
#     --seed 5 \
#     --lr 0.001 \
#     --lrmuon 0.01\
#     --power 0.25 \
#     --batch_size 256 \
#     --total_batch_size 512 \
#     --num_training_steps 10000 \
#     --warmup_steps 1000 \
#     --weight_decay 0.1 \
#     --dtype bfloat16 \
#     --eval_every 1000 \
#     --wandb_name muonG-60M-lr0.001-lrmuon0.01-WD0.1-step10000-repeat1 \
#     --target_eval_tokens 10_000_000 \
#     --save_every 10000








# export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
# export CUDA_VISIBLE_DEVICES=0,3
# torchrun --nproc_per_node=2 --master_port=21178 --master_addr=localhost torchrun_main_muon_test.py \
#     --model_config configs/llama_60m.json \
#     --optimizer muon \
#     --seed 5 \
#     --lr 0.001 \
#     --lrmuon 0.01\
#     --power 0.25 \
#     --batch_size 256 \
#     --total_batch_size 512 \
#     --num_training_steps 20000 \
#     --warmup_steps 2000 \
#     --weight_decay 0.1 \
#     --dtype bfloat16 \
#     --eval_every 1000 \
#     --wandb_name muon-60M-lr0.001-lrmuon0.01-WD0.1-repeat1 \
#     --target_eval_tokens 10_000_000 \
#     --save_every 10000


# export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
# export CUDA_VISIBLE_DEVICES=0,3
# torchrun --nproc_per_node=2 --master_port=22107 --master_addr=localhost torchrun_main_muon_test.py \
#     --model_config configs/llama_60m.json \
#     --optimizer muon_generalized \
#     --seed 5 \
#     --lr 0.001 \
#     --lrmuon 0.01\
#     --power 0.25 \
#     --batch_size 256 \
#     --total_batch_size 512 \
#     --num_training_steps 20000 \
#     --warmup_steps 2000 \
#     --weight_decay 0.1 \
#     --dtype bfloat16 \
#     --eval_every 1000 \
#     --wandb_name muonG-60M-lr0.001-lrmuon0.01-WD0.1-repeat1 \
#     --target_eval_tokens 10_000_000 \
#     --save_every 10000




export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
export CUDA_VISIBLE_DEVICES=0,3
torchrun --nproc_per_node=2 --master_port=21178 --master_addr=localhost torchrun_main_muon_test.py \
    --model_config configs/llama_60m.json \
    --optimizer muon \
    --seed 5 \
    --lr 0.001 \
    --lrmuon 0.02\
    --power 0.25 \
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps 30000 \
    --warmup_steps 3000 \
    --weight_decay 0.1 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --wandb_name muon-60M-lr0.001-lrmuon0.02-WD0.1-step30000-repeat1 \
    --target_eval_tokens 10_000_000 \
    --save_every 10000


export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
export CUDA_VISIBLE_DEVICES=0,3
torchrun --nproc_per_node=2 --master_port=22107 --master_addr=localhost torchrun_main_muon_test.py \
    --model_config configs/llama_60m.json \
    --optimizer muon_generalized \
    --seed 5 \
    --lr 0.001 \
    --lrmuon 0.02\
    --power 0.25 \
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps 30000 \
    --warmup_steps 3000 \
    --weight_decay 0.1 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --wandb_name muonG-60M-lr0.001-lrmuon0.02-WD0.1-step30000-repeat1 \
    --target_eval_tokens 10_000_000 \
    --save_every 10000







export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
export CUDA_VISIBLE_DEVICES=0,3
torchrun --nproc_per_node=2 --master_port=21178 --master_addr=localhost torchrun_main_muon_test.py \
    --model_config configs/llama_60m.json \
    --optimizer muon \
    --seed 5 \
    --lr 0.001 \
    --lrmuon 0.03\
    --power 0.25 \
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps 30000 \
    --warmup_steps 3000 \
    --weight_decay 0.1 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --wandb_name muon-60M-lr0.001-lrmuon0.03-WD0.1-step30000-repeat1 \
    --target_eval_tokens 10_000_000 \
    --save_every 10000


export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
export CUDA_VISIBLE_DEVICES=0,3
torchrun --nproc_per_node=2 --master_port=22107 --master_addr=localhost torchrun_main_muon_test.py \
    --model_config configs/llama_60m.json \
    --optimizer muon_generalized \
    --seed 5 \
    --lr 0.001 \
    --lrmuon 0.03\
    --power 0.25 \
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps 30000 \
    --warmup_steps 3000 \
    --weight_decay 0.1 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --wandb_name muonG-60M-lr0.001-lrmuon0.03-WD0.1-step30000-repeat1 \
    --target_eval_tokens 10_000_000 \
    --save_every 10000







export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
export CUDA_VISIBLE_DEVICES=0,3
torchrun --nproc_per_node=2 --master_port=21178 --master_addr=localhost torchrun_main_muon_test.py \
    --model_config configs/llama_60m.json \
    --optimizer muon \
    --seed 5 \
    --lr 0.001 \
    --lrmuon 0.04\
    --power 0.25 \
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps 30000 \
    --warmup_steps 3000 \
    --weight_decay 0.1 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --wandb_name muon-60M-lr0.001-lrmuon0.04-WD0.1-step30000-repeat1 \
    --target_eval_tokens 10_000_000 \
    --save_every 10000


export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
export CUDA_VISIBLE_DEVICES=0,3
torchrun --nproc_per_node=2 --master_port=22107 --master_addr=localhost torchrun_main_muon_test.py \
    --model_config configs/llama_60m.json \
    --optimizer muon_generalized \
    --seed 5 \
    --lr 0.001 \
    --lrmuon 0.04\
    --power 0.25 \
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps 30000 \
    --warmup_steps 3000 \
    --weight_decay 0.1 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --wandb_name muonG-60M-lr0.001-lrmuon0.04-WD0.1-step30000-repeat1 \
    --target_eval_tokens 10_000_000 \
    --save_every 10000







export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
export CUDA_VISIBLE_DEVICES=0,3
torchrun --nproc_per_node=2 --master_port=21178 --master_addr=localhost torchrun_main_muon_test.py \
    --model_config configs/llama_60m.json \
    --optimizer muon \
    --seed 5 \
    --lr 0.001 \
    --lrmuon 0.01\
    --power 0.25 \
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps 30000 \
    --warmup_steps 3000 \
    --weight_decay 0.1 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --wandb_name muon-60M-lr0.001-lrmuon0.01-WD0.1-step30000-repeat1 \
    --target_eval_tokens 10_000_000 \
    --save_every 10000


export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
export CUDA_VISIBLE_DEVICES=0,3
torchrun --nproc_per_node=2 --master_port=22107 --master_addr=localhost torchrun_main_muon_test.py \
    --model_config configs/llama_60m.json \
    --optimizer muon_generalized \
    --seed 5 \
    --lr 0.001 \
    --lrmuon 0.01\
    --power 0.25 \
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps 30000 \
    --warmup_steps 3000 \
    --weight_decay 0.1 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --wandb_name muonG-60M-lr0.001-lrmuon0.01-WD0.1-step30000-repeat1 \
    --target_eval_tokens 10_000_000 \
    --save_every 10000