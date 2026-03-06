export WANDB_API_KEY='Your_WandB_API_Key_Here'
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 --master_port=20119 --master_addr=localhost torchrun_main_HTMuon.py \
    --model_config configs/llama_1b.json \
    --optimizer htmuon_interval \
    --seed 5 \
    --lr 0.001 \
    --lrmuon 5e-3\
    --power 0.125 \
    --interval 5 \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 90000 \
    --warmup_steps 9000 \
    --weight_decay 0.1\
    --dtype bfloat16 \
    --eval_every 1000 \
    --wandb_name 'Your_WandB_Name_Here' \
    --target_eval_tokens 10_000_000 \
    --save_every 10000