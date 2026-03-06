# HTMuon: Improving Muon via Heavy-Tailed Spectral Correction




## Installation

### Setup

Our repository is built on top of [Galore](https://github.com/jiaweizzhao/GaLore), [MARS](https://github.com/AGI-Arena/MARS) and [AlphaDecay](https://github.com/hed-ucas/AlphaDecay). You can configure the environment using the following command lines:
```bash
conda create -n htmuon python=3.9 -y
conda activate htmuon
conda install -r requirements
```

### Prepare Dataset

We utilized the publicly available C4 dataset, OpenWebText dataset, CIFAR-10, CIFAR-100 and ImageNet-1K dataset, all of which can be accessed and downloaded from their respective official websites.

## Usage

#### Pretraining LLama-135M on C4
```bash
export WANDB_API_KEY='Your_WandB_API_Key_Here'
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 --master_port=20119 --master_addr=localhost torchrun_main_HTMuon.py \
    --model_config configs/llama_350m.json \
    --optimizer muon \
    --seed 5 \
    --lr 0.001 \
    --lrmuon 5e-3\
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 60000 \
    --warmup_steps 6000 \
    --weight_decay 0.1\
    --dtype bfloat16 \
    --eval_every 1000 \
    --wandb_name 'Your_WandB_Name_Here' \
    --target_eval_tokens 10_000_000 \
    --save_every 10000
```


### Acknowledgement
This repository is build upon the [Galore](https://github.com/jiaweizzhao/GaLore),[MARS](https://github.com/AGI-Arena/MARS)  and [AlphaDecay](https://github.com/hed-ucas/AlphaDecay) repositories. Thanks for their great work!

