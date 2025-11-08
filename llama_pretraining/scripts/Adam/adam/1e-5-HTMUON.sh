export WANDB_API_KEY='Your_WandB_API_Key_Here'
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 --master_port=20109 --master_addr=localhost torchrun_main_htmuon_v2.py \
    --model_config configs/llama_130m.json \
    --optimizer mlast_norg \
    --seed 5 \
    --lr 0.001 \
    --lrmuon 0.03\
    --power 0.25 \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 20000 \
    --warmup_steps 2000 \
    --weight_decay 0.1 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --wandb_name mlastnormuong-130M-lr0.001-lrmuon0.03-WD0.1-step20000-repeat1 \
    --target_eval_tokens 10_000_000 \
    --save_every 10000


export WANDB_API_KEY='Your_WandB_API_Key_Here'
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 --master_port=20109 --master_addr=localhost torchrun_main_htmuon_v2.py \
    --model_config configs/llama_130m.json \
    --optimizer mlast_g \
    --seed 5 \
    --lr 0.001 \
    --lrmuon 0.03\
    --power 0.25 \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 20000 \
    --warmup_steps 2000 \
    --weight_decay 0.1 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --wandb_name mlastmuong-130M-lr0.001-lrmuon0.03-WD0.1-step20000-repeat1 \
    --target_eval_tokens 10_000_000 \
    --save_every 10000










export WANDB_API_KEY='Your_WandB_API_Key_Here'
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 --master_port=20109 --master_addr=localhost torchrun_main_htmuon.py \
    --model_config configs/llama_130m.json \
    --optimizer muon_generalized \
    --seed 5 \
    --lr 0.001 \
    --lrmuon 0.03\
    --power 0.25 \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 20000 \
    --warmup_steps 2000 \
    --weight_decay 0.1 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --wandb_name muonG-130M-lr0.001-lrmuon0.03-WD0.1-step20000-repeat1 \
    --target_eval_tokens 10_000_000 \
    --save_every 10000


export WANDB_API_KEY='Your_WandB_API_Key_Here'
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 --master_port=21110 --master_addr=localhost torchrun_main_htmuon.py \
    --model_config configs/llama_130m.json \
    --optimizer muon \
    --seed 5 \
    --lr 0.001 \
    --lrmuon 0.03\
    --power 0.25 \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 20000 \
    --warmup_steps 2000 \
    --weight_decay 0.1 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --wandb_name muon-130M-lr0.001-lrmuon0.03-WD0.1-step20000-repeat1 \
    --target_eval_tokens 10_000_000 \
    --save_every 10000










export WANDB_API_KEY='Your_WandB_API_Key_Here'
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2  --master_port=20112 --master_addr=localhost torchrun_main_htmuon.py \
    --model_config configs/llama_60m.json \
    --optimizer muon_generalized \
    --seed 5 \
    --lr 0.001 \
    --lrmuon 0.03\
    --power 0.25 \
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps 20000 \
    --warmup_steps 2000 \
    --weight_decay 0.1 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --wandb_name muonG-60M-lr0.001-lrmuon0.03-WD0.1-step20000-repeat1 \
    --target_eval_tokens 10_000_000 \
    --save_every 10000


export WANDB_API_KEY='Your_WandB_API_Key_Here'
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 --master_port=21119 --master_addr=localhost torchrun_main_htmuon.py \
    --model_config configs/llama_60m.json \
    --optimizer muon \
    --seed 5 \
    --lr 0.001 \
    --lrmuon 0.03\
    --power 0.25 \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 20000 \
    --warmup_steps 2000 \
    --weight_decay 0.1 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --wandb_name muon-60M-lr0.001-lrmuon0.03-WD0.1-step20000-repeat1 \
    --target_eval_tokens 10_000_000 \
    --save_every 10000




export WANDB_API_KEY='Your_WandB_API_Key_Here'
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2  --master_port=20112 --master_addr=localhost torchrun_main_htmuon.py \
    --model_config configs/llama_60m.json \
    --optimizer muon_generalized \
    --seed 5 \
    --lr 0.001 \
    --lrmuon 0.01\
    --power 0.25 \
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_steps 1000 \
    --weight_decay 0.1 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --wandb_name muonG-60M-lr0.001-lrmuon0.01-WD0.1-step10000-repeat1 \
    --target_eval_tokens 10_000_000 \
    --save_every 10000


export WANDB_API_KEY='Your_WandB_API_Key_Here'
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 --master_port=21119 --master_addr=localhost torchrun_main_htmuon.py \
    --model_config configs/llama_60m.json \
    --optimizer muon \
    --seed 5 \
    --lr 0.001 \
    --lrmuon 0.01\
    --power 0.25 \
    --batch_size 256 \
    --total_batch_size 512 \
    --num_training_steps 10000 \
    --warmup_steps 1000 \
    --weight_decay 0.1 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --wandb_name muon-60M-lr0.001-lrmuon0.01-WD0.1-step10000-repeat1 \
    --target_eval_tokens 10_000_000 \
    --save_every 10000
