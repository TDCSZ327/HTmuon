import torch
import inspect
import os
from adamuon import AdaMuon

def configure_optimizers(params, weight_decay, learning_rate, lrmuon,betas=(0.9, 0.95), device_type='cuda'):
    param_dict = {pn: p for pn, p in params}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        
    nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
    # AdaMuon is not suitable for embedding and head layer, same as Muon.
    # Modify the Name.
    twoD_params = [p for n, p in param_dict.items() if p.dim() >= 2 and ('wte' not in n and 'lm_head' not in n)]
    decay_params = [p for n, p in param_dict.items() if 'wte' in n or 'lm_head' in n]
        
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    num_twoD_params = sum(p.numel() for p in twoD_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    print(f"num 2D parameter tensors: {len(twoD_params)}, with {num_twoD_params:,} parameters")
        
    # Adam
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

    # AdaMuon
    try:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
    except:
        rank = 0
        world_size = 1

    optimizer1 = AdaMuon(twoD_params, lr=lrmuon, momentum=0.95, rank=rank, world_size=world_size, weight_decay=weight_decay)
    print(f"using AdaMuon and AdamW")

    return [optimizer, optimizer1]