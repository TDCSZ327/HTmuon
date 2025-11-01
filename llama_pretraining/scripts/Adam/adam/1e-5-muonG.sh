#!/bin/bash
# bash hed_scripts/60M/adam/adam_60M_lr1e_3_WD1e_5.sh


# export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
# export CUDA_VISIBLE_DEVICES=1,2
# torchrun --nproc_per_node=2 --master_port=20102 --master_addr=localhost torchrun_main_muon_test.py \
#     --model_config configs/llama_60m.json \
#     --optimizer muon_generalized \
#     --seed 5 \
#     --lr 0.001 \
#     --lrmuon 0.02\
#     --power 0.25 \
#     --batch_size 256 \
#     --total_batch_size 512 \
#     --num_training_steps 20000 \
#     --warmup_steps 2000 \
#     --weight_decay 0.1 \
#     --dtype bfloat16 \
#     --eval_every 1000 \
#     --wandb_name muonG-60M-lr0.001-lrmuon0.02-WD0.1-repeat1 \
#     --target_eval_tokens 10_000_000 \
#     --save_every 10000

# export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
# export CUDA_VISIBLE_DEVICES=1,2
# torchrun --nproc_per_node=2 --master_port=20102 --master_addr=localhost torchrun_main_muon_test.py \
#     --model_config configs/llama_60m.json \
#     --optimizer muon_generalized \
#     --seed 5 \
#     --lr 0.001 \
#     --lrmuon 0.02\
#     --power 0.25 \
#     --batch_size 256 \
#     --total_batch_size 512 \
#     --num_training_steps 20000 \
#     --warmup_steps 2000 \
#     --weight_decay 0.1 \
#     --dtype bfloat16 \
#     --eval_every 1000 \
#     --wandb_name muonG-60M-lr0.001-lrmuon0.02-WD0.1-repeat1 \
#     --target_eval_tokens 10_000_000 \
#     --save_every 10000



# export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
# export CUDA_VISIBLE_DEVICES=1,4,5,6
# torchrun --nproc_per_node=4 --master_port=21115 --master_addr=localhost torchrun_main_muon_test.py \
#     --model_config configs/llama_130m.json \
#     --optimizer muon \
#     --seed 5 \
#     --lr 0.001 \
#     --lrmuon 0.02\
#     --power 0.25 \
#     --batch_size 128 \
#     --total_batch_size 512 \
#     --num_training_steps 20000 \
#     --warmup_steps 2000 \
#     --weight_decay 0.1 \
#     --dtype bfloat16 \
#     --eval_every 1000 \
#     --wandb_name muon-130M-lr0.001-lrmuon0.02-WD0.1-repeat1 \
#     --target_eval_tokens 10_000_000 \
#     --save_every 10000

# export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
# export CUDA_VISIBLE_DEVICES=1,4,5,6
# torchrun --nproc_per_node=4 --master_port=20102 --master_addr=localhost torchrun_main_muon_test.py \
#     --model_config configs/llama_130m.json \
#     --optimizer muon_generalized \
#     --seed 5 \
#     --lr 0.001 \
#     --lrmuon 0.02\
#     --power 0.25 \
#     --batch_size 128 \
#     --total_batch_size 512 \
#     --num_training_steps 20000 \
#     --warmup_steps 2000 \
#     --weight_decay 0.1 \
#     --dtype bfloat16 \
#     --eval_every 1000 \
#     --wandb_name muonG-130M-lr0.001-lrmuon0.02-WD0.1-repeat1 \
#     --target_eval_tokens 10_000_000 \
#     --save_every 10000

# export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
# export CUDA_VISIBLE_DEVICES=1,4,5,6
# torchrun --nproc_per_node=4 --master_port=21115 --master_addr=localhost torchrun_main_muon_test.py \
#     --model_config configs/llama_130m.json \
#     --optimizer muon \
#     --seed 5 \
#     --lr 0.001 \
#     --lrmuon 0.03\
#     --power 0.25 \
#     --batch_size 128 \
#     --total_batch_size 512 \
#     --num_training_steps 20000 \
#     --warmup_steps 2000 \
#     --weight_decay 0.1 \
#     --dtype bfloat16 \
#     --eval_every 1000 \
#     --wandb_name muon-130M-lr0.001-lrmuon0.03-WD0.1-repeat1 \
#     --target_eval_tokens 10_000_000 \
#     --save_every 10000

# export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
# export CUDA_VISIBLE_DEVICES=1,4,5,6
# torchrun --nproc_per_node=4 --master_port=20102 --master_addr=localhost torchrun_main_muon_test.py \
#     --model_config configs/llama_130m.json \
#     --optimizer muon_generalized \
#     --seed 5 \
#     --lr 0.001 \
#     --lrmuon 0.03\
#     --power 0.25 \
#     --batch_size 128 \
#     --total_batch_size 512 \
#     --num_training_steps 20000 \
#     --warmup_steps 2000 \
#     --weight_decay 0.1 \
#     --dtype bfloat16 \
#     --eval_every 1000 \
#     --wandb_name muonG-130M-lr0.001-lrmuon0.03-WD0.1-repeat1 \
#     --target_eval_tokens 10_000_000 \
#     --save_every 10000


# export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
# export CUDA_VISIBLE_DEVICES=1,4,5,6
# torchrun --nproc_per_node=4 --master_port=21115 --master_addr=localhost torchrun_main_muon_test.py \
#     --model_config configs/llama_130m.json \
#     --optimizer muon \
#     --seed 5 \
#     --lr 0.001 \
#     --lrmuon 0.01\
#     --power 0.25 \
#     --batch_size 128 \
#     --total_batch_size 512 \
#     --num_training_steps 20000 \
#     --warmup_steps 2000 \
#     --weight_decay 0.1 \
#     --dtype bfloat16 \
#     --eval_every 1000 \
#     --wandb_name muon-130M-lr0.001-lrmuon0.01-WD0.1-repeat1 \
#     --target_eval_tokens 10_000_000 \
#     --save_every 10000

# export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
# export CUDA_VISIBLE_DEVICES=1,4,5,6
# torchrun --nproc_per_node=4 --master_port=20102 --master_addr=localhost torchrun_main_muon_test.py \
#     --model_config configs/llama_130m.json \
#     --optimizer muon_generalized \
#     --seed 5 \
#     --lr 0.001 \
#     --lrmuon 0.01\
#     --power 0.25 \
#     --batch_size 128 \
#     --total_batch_size 512 \
#     --num_training_steps 20000 \
#     --warmup_steps 2000 \
#     --weight_decay 0.1 \
#     --dtype bfloat16 \
#     --eval_every 1000 \
#     --wandb_name muonG-130M-lr0.001-lrmuon0.01-WD0.1-repeat1 \
#     --target_eval_tokens 10_000_000 \
#     --save_every 10000



# export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
# export CUDA_VISIBLE_DEVICES=1,4,5,6
# torchrun --nproc_per_node=4 --master_port=21115 --master_addr=localhost torchrun_main_muon_test.py \
#     --model_config configs/llama_130m.json \
#     --optimizer muon \
#     --seed 5 \
#     --lr 0.001 \
#     --lrmuon 0.04\
#     --power 0.25 \
#     --batch_size 128 \
#     --total_batch_size 512 \
#     --num_training_steps 20000 \
#     --warmup_steps 2000 \
#     --weight_decay 0.1 \
#     --dtype bfloat16 \
#     --eval_every 1000 \
#     --wandb_name muon-130M-lr0.001-lrmuon0.04-WD0.1-repeat1 \
#     --target_eval_tokens 10_000_000 \
#     --save_every 10000

# export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
# export CUDA_VISIBLE_DEVICES=1,4,5,6
# torchrun --nproc_per_node=4 --master_port=20102 --master_addr=localhost torchrun_main_muon_test.py \
#     --model_config configs/llama_130m.json \
#     --optimizer muon_generalized \
#     --seed 5 \
#     --lr 0.001 \
#     --lrmuon 0.04\
#     --power 0.25 \
#     --batch_size 128 \
#     --total_batch_size 512 \
#     --num_training_steps 20000 \
#     --warmup_steps 2000 \
#     --weight_decay 0.1 \
#     --dtype bfloat16 \
#     --eval_every 1000 \
#     --wandb_name muonG-130M-lr0.001-lrmuon0.04-WD0.1-repeat1 \
#     --target_eval_tokens 10_000_000 \
#     --save_every 10000



# export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# torchrun --nproc_per_node=8 --master_port=21115 --master_addr=localhost torchrun_main_muon_test.py \
#     --model_config configs/llama_350m.json \
#     --optimizer muon \
#     --seed 5 \
#     --lr 0.001 \
#     --lrmuon 0.02\
#     --power 0.25 \
#     --batch_size 64 \
#     --total_batch_size 512 \
#     --num_training_steps 60000 \
#     --warmup_steps 6000 \
#     --weight_decay 0.1 \
#     --dtype bfloat16 \
#     --eval_every 1000 \
#     --wandb_name muon-350M-lr0.001-lrmuon0.02-WD0.1-step60000-repeat1 \
#     --target_eval_tokens 10_000_000 \
#     --save_every 10000

# export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# torchrun --nproc_per_node=8 --master_port=20102 --master_addr=localhost torchrun_main_muon_test.py \
#     --model_config configs/llama_350m.json \
#     --optimizer muon_generalized \
#     --seed 5 \
#     --lr 0.001 \
#     --lrmuon 0.02\
#     --power 0.25 \
#     --batch_size 64 \
#     --total_batch_size 512 \
#     --num_training_steps 60000 \
#     --warmup_steps 6000 \
#     --weight_decay 0.1 \
#     --dtype bfloat16 \
#     --eval_every 1000 \
#     --wandb_name muonG-350M-lr0.001-lrmuon0.02-WD0.1-step60000-repeat1 \
#     --target_eval_tokens 10_000_000 \
#     --save_every 10000

# export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# torchrun --nproc_per_node=8 --master_port=21115 --master_addr=localhost torchrun_main_muon_test.py \
#     --model_config configs/llama_350m.json \
#     --optimizer muon \
#     --seed 5 \
#     --lr 0.001 \
#     --lrmuon 0.03\
#     --power 0.25 \
#     --batch_size 64 \
#     --total_batch_size 512 \
#     --num_training_steps 60000 \
#     --warmup_steps 6000 \
#     --weight_decay 0.1 \
#     --dtype bfloat16 \
#     --eval_every 1000 \
#     --wandb_name muon-350M-lr0.001-lrmuon0.03-WD0.1-step60000-repeat1 \
#     --target_eval_tokens 10_000_000 \
#     --save_every 10000

# export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# torchrun --nproc_per_node=8 --master_port=20102 --master_addr=localhost torchrun_main_muon_test.py \
#     --model_config configs/llama_350m.json \
#     --optimizer muon_generalized \
#     --seed 5 \
#     --lr 0.001 \
#     --lrmuon 0.03\
#     --power 0.25 \
#     --batch_size 64 \
#     --total_batch_size 512 \
#     --num_training_steps 60000 \
#     --warmup_steps 6000 \
#     --weight_decay 0.1 \
#     --dtype bfloat16 \
#     --eval_every 1000 \
#     --wandb_name muonG-350M-lr0.001-lrmuon0.03-WD0.1-step60000-repeat1 \
#     --target_eval_tokens 10_000_000 \
#     --save_every 10000



# export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# torchrun --nproc_per_node=8 --master_port=21115 --master_addr=localhost torchrun_main_muon_test.py \
#     --model_config configs/llama_350m.json \
#     --optimizer muon \
#     --seed 5 \
#     --lr 0.001 \
#     --lrmuon 0.02\
#     --power 0.25 \
#     --batch_size 128 \
#     --total_batch_size 1024 \
#     --num_training_steps 30000 \
#     --warmup_steps 3000 \
#     --weight_decay 0.1 \
#     --dtype bfloat16 \
#     --eval_every 1000 \
#     --wandb_name muon-350M-lr0.001-lrmuon0.02-WD0.1-step30000-repeat1 \
#     --target_eval_tokens 10_000_000 \
#     --save_every 10000

# export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# torchrun --nproc_per_node=8 --master_port=20102 --master_addr=localhost torchrun_main_muon_test.py \
#     --model_config configs/llama_350m.json \
#     --optimizer muon_generalized \
#     --seed 5 \
#     --lr 0.001 \
#     --lrmuon 0.02\
#     --power 0.25 \
#     --batch_size 128 \
#     --total_batch_size 1024\
#     --num_training_steps 30000 \
#     --warmup_steps 3000 \
#     --weight_decay 0.1 \
#     --dtype bfloat16 \
#     --eval_every 1000 \
#     --wandb_name muonG-350M-lr0.001-lrmuon0.02-WD0.1-step30000-repeat1 \
#     --target_eval_tokens 10_000_000 \
#     --save_every 10000

# export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# torchrun --nproc_per_node=8 --master_port=21115 --master_addr=localhost torchrun_main_muon_test.py \
#     --model_config configs/llama_350m.json \
#     --optimizer muon \
#     --seed 5 \
#     --lr 0.001 \
#     --lrmuon 0.03\
#     --power 0.25 \
#     --batch_size 128 \
#     --total_batch_size 1024 \
#     --num_training_steps 30000 \
#     --warmup_steps 3000 \
#     --weight_decay 0.1 \
#     --dtype bfloat16 \
#     --eval_every 1000 \
#     --wandb_name muon-350M-lr0.001-lrmuon0.03-WD0.1-step30000-repeat1 \
#     --target_eval_tokens 10_000_000 \
#     --save_every 10000

# export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# torchrun --nproc_per_node=8 --master_port=20102 --master_addr=localhost torchrun_main_muon_test.py \
#     --model_config configs/llama_350m.json \
#     --optimizer muon_generalized \
#     --seed 5 \
#     --lr 0.001 \
#     --lrmuon 0.03\
#     --power 0.25 \
#     --batch_size 128 \
#     --total_batch_size 1024 \
#     --num_training_steps 30000 \
#     --warmup_steps 3000 \
#     --weight_decay 0.1 \
#     --dtype bfloat16 \
#     --eval_every 1000 \
#     --wandb_name muonG-350M-lr0.001-lrmuon0.03-WD0.1-step30000-repeat1 \
#     --target_eval_tokens 10_000_000 \
#     --save_every 10000





# export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
# export CUDA_VISIBLE_DEVICES=1,4,5,6
# torchrun --nproc_per_node=4 --master_port=21115 --master_addr=localhost torchrun_main_muon_test.py \
#     --model_config configs/llama_130m.json \
#     --optimizer muon \
#     --seed 5 \
#     --lr 0.001 \
#     --lrmuon 0.02\
#     --power 0.25 \
#     --batch_size 128 \
#     --total_batch_size 512 \
#     --num_training_steps 30000 \
#     --warmup_steps 3000 \
#     --weight_decay 0.1 \
#     --dtype bfloat16 \
#     --eval_every 1000 \
#     --wandb_name muon-130M-lr0.001-lrmuon0.02-WD0.1-step30000-repeat1 \
#     --target_eval_tokens 10_000_000 \
#     --save_every 10000

# export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
# export CUDA_VISIBLE_DEVICES=1,4,5,6
# torchrun --nproc_per_node=4 --master_port=20102 --master_addr=localhost torchrun_main_muon_test.py \
#     --model_config configs/llama_130m.json \
#     --optimizer muon_generalized \
#     --seed 5 \
#     --lr 0.001 \
#     --lrmuon 0.02\
#     --power 0.25 \
#     --batch_size 128 \
#     --total_batch_size 512 \
#     --num_training_steps 30000 \
#     --warmup_steps 3000 \
#     --weight_decay 0.1 \
#     --dtype bfloat16 \
#     --eval_every 1000 \
#     --wandb_name muonG-130M-lr0.001-lrmuon0.02-WD0.1-step30000-repeat1 \
#     --target_eval_tokens 10_000_000 \
#     --save_every 10000




# export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
# export CUDA_VISIBLE_DEVICES=1,4,5,6
# torchrun --nproc_per_node=4 --master_port=21116 --master_addr=localhost torchrun_main_muon_test.py \
#     --model_config configs/llama_130m.json \
#     --optimizer muon \
#     --seed 5 \
#     --lr 0.001 \
#     --lrmuon 0.03\
#     --power 0.25 \
#     --batch_size 128 \
#     --total_batch_size 512 \
#     --num_training_steps 30000 \
#     --warmup_steps 3000 \
#     --weight_decay 0.1 \
#     --dtype bfloat16 \
#     --eval_every 1000 \
#     --wandb_name muon-130M-lr0.001-lrmuon0.03-WD0.1-step30000-repeat1 \
#     --target_eval_tokens 10_000_000 \
#     --save_every 10000

# export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
# export CUDA_VISIBLE_DEVICES=1,4,5,6
# torchrun --nproc_per_node=4 --master_port=20107 --master_addr=localhost torchrun_main_muon_test.py \
#     --model_config configs/llama_130m.json \
#     --optimizer muon_generalized \
#     --seed 5 \
#     --lr 0.001 \
#     --lrmuon 0.03\
#     --power 0.25 \
#     --batch_size 128 \
#     --total_batch_size 512 \
#     --num_training_steps 30000 \
#     --warmup_steps 3000 \
#     --weight_decay 0.1 \
#     --dtype bfloat16 \
#     --eval_every 1000 \
#     --wandb_name muonG-130M-lr0.001-lrmuon0.03-WD0.1-step30000-repeat1 \
#     --target_eval_tokens 10_000_000 \
#     --save_every 10000


# export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
# export CUDA_VISIBLE_DEVICES=1,4,5,6
# torchrun --nproc_per_node=4 --master_port=21118 --master_addr=localhost torchrun_main_muon_test.py \
#     --model_config configs/llama_130m.json \
#     --optimizer muon \
#     --seed 5 \
#     --lr 0.001 \
#     --lrmuon 0.01\
#     --power 0.25 \
#     --batch_size 128 \
#     --total_batch_size 512 \
#     --num_training_steps 30000 \
#     --warmup_steps 3000 \
#     --weight_decay 0.1 \
#     --dtype bfloat16 \
#     --eval_every 1000 \
#     --wandb_name muon-130M-lr0.001-lrmuon0.01-WD0.1-step30000-repeat1 \
#     --target_eval_tokens 10_000_000 \
#     --save_every 10000

# export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
# export CUDA_VISIBLE_DEVICES=1,4,5,6
# torchrun --nproc_per_node=4 --master_port=20109 --master_addr=localhost torchrun_main_muon_test.py \
#     --model_config configs/llama_130m.json \
#     --optimizer muon_generalized \
#     --seed 5 \
#     --lr 0.001 \
#     --lrmuon 0.01\
#     --power 0.25 \
#     --batch_size 128 \
#     --total_batch_size 512 \
#     --num_training_steps 30000 \
#     --warmup_steps 3000 \
#     --weight_decay 0.1 \
#     --dtype bfloat16 \
#     --eval_every 1000 \
#     --wandb_name muonG-130M-lr0.001-lrmuon0.01-WD0.1-step30000-repeat1 \
#     --target_eval_tokens 10_000_000 \
#     --save_every 10000


# export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
# export CUDA_VISIBLE_DEVICES=1,4,5,6
# torchrun --nproc_per_node=4 --master_port=21110 --master_addr=localhost torchrun_main_muon_test.py \
#     --model_config configs/llama_130m.json \
#     --optimizer muon \
#     --seed 5 \
#     --lr 0.001 \
#     --lrmuon 0.04\
#     --power 0.25 \
#     --batch_size 128 \
#     --total_batch_size 512 \
#     --num_training_steps 30000 \
#     --warmup_steps 3000 \
#     --weight_decay 0.1 \
#     --dtype bfloat16 \
#     --eval_every 1000 \
#     --wandb_name muon-130M-lr0.001-lrmuon0.04-WD0.1-step30000-repeat1 \
#     --target_eval_tokens 10_000_000 \
#     --save_every 10000

# export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
# export CUDA_VISIBLE_DEVICES=1,4,5,6
# torchrun --nproc_per_node=4 --master_port=20104 --master_addr=localhost torchrun_main_muon_test.py \
#     --model_config configs/llama_130m.json \
#     --optimizer muon_generalized \
#     --seed 5 \
#     --lr 0.001 \
#     --lrmuon 0.04\
#     --power 0.25 \
#     --batch_size 128 \
#     --total_batch_size 512 \
#     --num_training_steps 30000 \
#     --warmup_steps 3000 \
#     --weight_decay 0.1 \
#     --dtype bfloat16 \
#     --eval_every 1000 \
#     --wandb_name muonG-130M-lr0.001-lrmuon0.04-WD0.1-step30000-repeat1 \
#     --target_eval_tokens 10_000_000 \
#     --save_every 10000

# export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
# export CUDA_VISIBLE_DEVICES=1,4,5,6
# torchrun --nproc_per_node=4 --master_port=20107 --master_addr=localhost torchrun_main_muon_test.py \
#     --model_config configs/llama_130m.json \
#     --optimizer adamw \
#     --seed 5 \
#     --lr 0.001 \
#     --lrmuon 0.03\
#     --power 0.25 \
#     --batch_size 128 \
#     --total_batch_size 512 \
#     --num_training_steps 20000 \
#     --warmup_steps 2000 \
#     --weight_decay 0.1 \
#     --dtype bfloat16 \
#     --eval_every 1000 \
#     --wandb_name adamw-130M-lr0.001-WD0.1-step20000-repeat1 \
#     --target_eval_tokens 10_000_000 \
#     --save_every 10000

# export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
# export CUDA_VISIBLE_DEVICES=1,4,5,6
# torchrun --nproc_per_node=4 --master_port=20109 --master_addr=localhost torchrun_main_muon_test.py \
#     --model_config configs/llama_130m.json \
#     --optimizer adam \
#     --seed 5 \
#     --lr 0.001 \
#     --lrmuon 0.03\
#     --power 0.25 \
#     --batch_size 128 \
#     --total_batch_size 512 \
#     --num_training_steps 20000 \
#     --warmup_steps 2000 \
#     --weight_decay 0.00001 \
#     --dtype bfloat16 \
#     --eval_every 1000 \
#     --wandb_name adam-130M-lr0.001-WD0.00001-step20000-repeat1 \
#     --target_eval_tokens 10_000_000 \
#     --save_every 10000

# export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
# export CUDA_VISIBLE_DEVICES=1,4,5,6
# torchrun --nproc_per_node=4 --master_port=20110 --master_addr=localhost torchrun_main_muon_test.py \
#     --model_config configs/llama_130m.json \
#     --optimizer muon_generalized \
#     --seed 5 \
#     --lr 0.001 \
#     --lrmuon 0.03\
#     --power 0.25 \
#     --batch_size 128 \
#     --total_batch_size 512 \
#     --num_training_steps 30000 \
#     --warmup_steps 3000 \
#     --weight_decay 0.1 \
#     --dtype bfloat16 \
#     --eval_every 1000 \
#     --wandb_name muonG-130M-lr0.001-lrmuon0.03-WD0.1-step30000-repeat1 \
#     --target_eval_tokens 10_000_000 \
#     --save_every 10000

# export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
# export CUDA_VISIBLE_DEVICES=1,4,5,6
# torchrun --nproc_per_node=4 --master_port=20107 --master_addr=localhost torchrun_main_muon_test.py \
#     --model_config configs/llama_130m.json \
#     --optimizer adamw \
#     --seed 5 \
#     --lr 0.001 \
#     --lrmuon 0.03\
#     --power 0.25 \
#     --batch_size 128 \
#     --total_batch_size 512 \
#     --num_training_steps 20000 \
#     --warmup_steps 2000 \
#     --weight_decay 0.1 \
#     --dtype bfloat16 \
#     --eval_every 1000 \
#     --wandb_name adamw-130M-lr0.001-WD0.1-step20000-repeat1 \
#     --target_eval_tokens 10_000_000 \
#     --save_every 10000


# export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
# export CUDA_VISIBLE_DEVICES=1,4
# torchrun --nproc_per_node=2 --master_port=22107 --master_addr=localhost torchrun_main_muon_test.py \
#     --model_config configs/llama_60m.json \
#     --optimizer adam \
#     --seed 5 \
#     --lr 0.001 \
#     --lrmuon 0.01\
#     --power 0.25 \
#     --batch_size 256 \
#     --total_batch_size 512 \
#     --num_training_steps 10000 \
#     --warmup_steps 1000 \
#     --weight_decay 0.00001 \
#     --dtype bfloat16 \
#     --eval_every 1000 \
#     --wandb_name adam-60M-lr0.001-WD0.00001-step10000-repeat1 \
#     --target_eval_tokens 10_000_000 \
#     --save_every 10000

# export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
# export CUDA_VISIBLE_DEVICES=1,4
# torchrun --nproc_per_node=2 --master_port=22108 --master_addr=localhost torchrun_main_muon_test.py \
#     --model_config configs/llama_60m.json \
#     --optimizer adamw \
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
#     --wandb_name adamw-60M-lr0.001-WD0.1-step10000-repeat1 \
#     --target_eval_tokens 10_000_000 \
#     --save_every 10000






# export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
# export CUDA_VISIBLE_DEVICES=0,1,4,6
# torchrun --nproc_per_node=4 --master_port=20107 --master_addr=localhost torchrun_main_muon_test.py \
#     --model_config configs/llama_130m.json \
#     --optimizer soap \
#     --seed 5 \
#     --lr 0.003 \
#     --lrmuon 0.03\
#     --power 0.25 \
#     --batch_size 128 \
#     --total_batch_size 512 \
#     --num_training_steps 20000 \
#     --warmup_steps 2000 \
#     --weight_decay 0.01 \
#     --dtype bfloat16 \
#     --eval_every 1000 \
#     --wandb_name soap-130M-lr0.003-WD0.01-step20000-repeat1 \
#     --target_eval_tokens 10_000_000 \
#     --save_every 10000












# export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
# export CUDA_VISIBLE_DEVICES=1,4
# torchrun --nproc_per_node=2 --master_port=22108 --master_addr=localhost torchrun_main_muon_test.py \
#     --model_config configs/llama_60m.json \
#     --optimizer soap \
#     --seed 5 \
#     --lr 0.003 \
#     --lrmuon 0.01\
#     --power 0.25 \
#     --batch_size 256 \
#     --total_batch_size 512 \
#     --num_training_steps 10000 \
#     --warmup_steps 1000 \
#     --weight_decay 0.01 \
#     --dtype bfloat16 \
#     --eval_every 1000 \
#     --wandb_name soap-60M-lr0.003-WD0.01-step10000-repeat1 \
#     --target_eval_tokens 10_000_000 \
#     --save_every 10000


# export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
# export CUDA_VISIBLE_DEVICES=1,4
# torchrun --nproc_per_node=2 --master_port=22108 --master_addr=localhost torchrun_main_muon_test.py \
#     --model_config configs/llama_60m.json \
#     --optimizer soap \
#     --seed 5 \
#     --lr 0.003 \
#     --lrmuon 0.01\
#     --power 0.25 \
#     --batch_size 256 \
#     --total_batch_size 512 \
#     --num_training_steps 20000 \
#     --warmup_steps 2000 \
#     --weight_decay 0.01 \
#     --dtype bfloat16 \
#     --eval_every 1000 \
#     --wandb_name soap-60M-lr0.003-WD0.01-step10000-repeat1 \
#     --target_eval_tokens 10_000_000 \
#     --save_every 10000




export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
export CUDA_VISIBLE_DEVICES=0,1,2,4
torchrun --nproc_per_node=4 --master_port=20107 --master_addr=localhost torchrun_main_muon_test.py \
    --model_config configs/llama_130m.json \
    --optimizer soap \
    --seed 5 \
    --lr 0.004 \
    --lrmuon 0.01\
    --power 0.25 \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 20000 \
    --warmup_steps 2000 \
    --weight_decay 0.01 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --wandb_name soap-130M-lr0.004-WD0.01-step20000-repeat1 \
    --target_eval_tokens 10_000_000 \
    --save_every 10000



export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
export CUDA_VISIBLE_DEVICES=0,1,2,4
torchrun --nproc_per_node=4 --master_port=20199 --master_addr=localhost torchrun_main_muon_test.py \
    --model_config configs/llama_130m.json \
    --optimizer soap \
    --seed 5 \
    --lr 0.002 \
    --lrmuon 0.01\
    --power 0.25 \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 20000 \
    --warmup_steps 2000 \
    --weight_decay 0.01 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --wandb_name soap-130M-lr0.002-WD0.01-step20000-repeat1 \
    --target_eval_tokens 10_000_000 \
    --save_every 10000









export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
export CUDA_VISIBLE_DEVICES=1,4
torchrun --nproc_per_node=2 --master_port=22108 --master_addr=localhost torchrun_main_muon_test.py \
    --model_config configs/llama_60m.json \
    --optimizer soap \
    --seed 5 \
    --lr 0.004 \
    --lrmuon 0.01\
    --power 0.25 \
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_steps 1000 \
    --weight_decay 0.01 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --wandb_name soap-60M-lr0.004-WD0.01-step10000-repeat1 \
    --target_eval_tokens 10_000_000 \
    --save_every 10000


export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
export CUDA_VISIBLE_DEVICES=1,4
torchrun --nproc_per_node=2 --master_port=22108 --master_addr=localhost torchrun_main_muon_test.py \
    --model_config configs/llama_60m.json \
    --optimizer soap \
    --seed 5 \
    --lr 0.004 \
    --lrmuon 0.01\
    --power 0.25 \
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps 20000 \
    --warmup_steps 2000 \
    --weight_decay 0.01 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --wandb_name soap-60M-lr0.004-WD0.01-step10000-repeat1 \
    --target_eval_tokens 10_000_000 \
    --save_every 10000





export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
export CUDA_VISIBLE_DEVICES=1,4
torchrun --nproc_per_node=2 --master_port=22108 --master_addr=localhost torchrun_main_muon_test.py \
    --model_config configs/llama_60m.json \
    --optimizer soap \
    --seed 5 \
    --lr 0.002 \
    --lrmuon 0.01\
    --power 0.25 \
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_steps 1000 \
    --weight_decay 0.01 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --wandb_name soap-60M-lr0.002-WD0.01-step10000-repeat1 \
    --target_eval_tokens 10_000_000 \
    --save_every 10000


export WANDB_API_KEY="c39a0e003988e2bd18ba31357cf0f9a1aab0245f"
export CUDA_VISIBLE_DEVICES=1,4
torchrun --nproc_per_node=2 --master_port=22108 --master_addr=localhost torchrun_main_muon_test.py \
    --model_config configs/llama_60m.json \
    --optimizer soap \
    --seed 5 \
    --lr 0.002 \
    --lrmuon 0.01\
    --power 0.25 \
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps 20000 \
    --warmup_steps 2000 \
    --weight_decay 0.01 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --wandb_name soap-60M-lr0.002-WD0.01-step10000-repeat1 \
    --target_eval_tokens 10_000_000 \
    --save_every 10000