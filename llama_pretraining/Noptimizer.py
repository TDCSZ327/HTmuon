
# All imports at the top
import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from abc import abstractmethod
import math

# Define AbstractOptimizer first
class AbstractOptimizer(Optimizer):
    def __init__(self, params, defaults):
        super().__init__(params, defaults)

    @abstractmethod
    def step(self, closure=None):
        """
        Performs a single optimization step (parameter update).
        To be implemented by subclasses.
        """
        pass

# NormGradOptimizer
class NormGradOptimizer(AbstractOptimizer):
    def __init__(self, params, lr=0.01, weight_decay=0.0):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self, closure=None):
        for group in self.param_groups:
            lr = group.get('lr', self.lr)
            weight_decay = group.get('weight_decay', self.weight_decay)
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if weight_decay > 0:
                    grad = grad + weight_decay * p.data
                if grad.dim() == 2:
                    normed = F.normalize(grad, p=2, dim=0)
                    p.data.add_(normed, alpha=-lr)
                elif grad.dim() == 1:
                    normed = F.normalize(grad, p=2, dim=0)
                    p.data.add_(normed, alpha=-lr)

# NormGradMomentumOptimizer
class NormGradMomentumOptimizer(AbstractOptimizer):
    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.state = {}

    def step(self, closure=None):
        for group in self.param_groups:
            lr = group.get('lr', self.lr)
            momentum = group.get('momentum', self.momentum)
            weight_decay = group.get('weight_decay', self.weight_decay)
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                param_state = self.state.setdefault(p, {})
                if 'momentum_buffer' not in param_state:
                    buf = torch.zeros_like(grad)
                else:
                    buf = param_state['momentum_buffer']
                buf = momentum * buf + grad
                if buf.dim() == 2:
                    normed = F.normalize(buf, p=2, dim=0)
                elif buf.dim() == 1:
                    normed = F.normalize(buf, p=2, dim=0)
                else:
                    normed = buf
                if weight_decay > 0:
                    normed = normed + weight_decay * p.data
                p.data.add_(normed, alpha=-lr)
                param_state['momentum_buffer'] = buf

# NormGradNesterovOptimizer
class NormGradNesterovOptimizer(AbstractOptimizer):
    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.state = {}

    def step(self, closure=None):
        for group in self.param_groups:
            lr = group.get('lr', self.lr)
            momentum = group.get('momentum', self.momentum)
            weight_decay = group.get('weight_decay', self.weight_decay)
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                param_state = self.state.setdefault(p, {})
                if weight_decay > 0:
                    grad = grad + weight_decay * p.data
                if 'momentum_buffer' not in param_state:
                    buf = torch.zeros_like(grad)
                else:
                    buf = param_state['momentum_buffer']
                buf = momentum * buf + grad
                nesterov_grad = grad + momentum * buf
                if nesterov_grad.dim() == 2:
                    normed = F.normalize(nesterov_grad, p=2, dim=0)
                elif nesterov_grad.dim() == 1:
                    normed = F.normalize(nesterov_grad, p=2, dim=0)
                else:
                    normed = nesterov_grad
                p.data.add_(normed, alpha=-lr)
                param_state['momentum_buffer'] = buf

# RNNPOptimizer
class RNNPOptimizer(AbstractOptimizer):
    def __init__(self, params, lr=0.01, momentum=0.9, beta=0.95, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8):
        defaults = dict(lr=lr, momentum=momentum, beta=beta, weight_decay=weight_decay, betas=betas, eps=eps)
        super().__init__(params, defaults)
        self.lr = lr
        self.momentum = momentum
        self.beta = beta
        self.weight_decay = weight_decay
        self.betas = betas
        self.eps = eps
        self.state = {}

    def step(self, closure=None):
        for group in self.param_groups:
            lr = group.get('lr', self.lr)
            momentum = group.get('momentum', self.momentum)
            beta = group.get('beta', self.beta)
            weight_decay = group.get('weight_decay', self.weight_decay)
            betas = group.get('betas', self.betas)
            eps = group.get('eps', self.eps)
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                param_state = self.state.setdefault(p, {})
                if grad.dim() == 2:
                    if 'momentum_buffer' not in param_state:
                        buf = torch.zeros_like(grad)
                    else:
                        buf = param_state['momentum_buffer']
                    buf.lerp_(grad, 1 - beta)
                    nesterov_buf = grad.lerp(buf, momentum)
                    normed = F.normalize(nesterov_buf, p=2, dim=1)
                    if weight_decay > 0:
                        p.data.mul_(1 - lr * weight_decay)
                    p.data.add_(normed, alpha=-lr)
                    param_state['momentum_buffer'] = buf
                elif grad.dim() == 1 or grad.dim() == 0:
                    if 'exp_avg' not in param_state:
                        param_state['exp_avg'] = torch.zeros_like(grad)
                        param_state['exp_avg_sq'] = torch.zeros_like(grad)
                        param_state['step'] = 0
                    exp_avg, exp_avg_sq = param_state['exp_avg'], param_state['exp_avg_sq']
                    param_state['step'] += 1
                    exp_avg.mul_(betas[0]).add_(grad, alpha=1-betas[0])
                    exp_avg_sq.mul_(betas[1]).addcmul_(grad, grad, value=1-betas[1])
                    bias_correction1 = 1 - betas[0] ** param_state['step']
                    bias_correction2 = 1 - betas[1] ** param_state['step']
                    step_size = lr * math.sqrt(bias_correction2) / bias_correction1
                    denom = exp_avg_sq.sqrt().add_(eps)
                    adam_update = exp_avg / denom
                    if weight_decay > 0:
                        p.data.mul_(1 - step_size * weight_decay)
                    p.data.add_(adam_update, alpha=-step_size)

    def distributed_step(self, closure=None):
        import torch.distributed as dist
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params = group['params']
            world_size = dist.get_world_size() if dist.is_initialized() else 1
            rank = dist.get_rank() if dist.is_initialized() else 0
            params_pad = params + [torch.empty_like(params[-1])] * (world_size - len(params) % world_size)


class NormGradOptimizer(AbstractOptimizer):
    def __init__(self, params, lr=0.01, weight_decay=0.0):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self, closure=None):
        for group in self.param_groups:
            lr = group.get('lr', self.lr)
            weight_decay = group.get('weight_decay', self.weight_decay)
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                # L2 regularization term
                if weight_decay > 0:
                    grad = grad + weight_decay * p.data
                # Matrix parameters
                if grad.dim() == 2:
                    normed = F.normalize(grad, p=2, dim=0)
                    p.data.add_(normed, alpha=-lr)
                elif grad.dim() == 1:
                    normed = F.normalize(grad, p=2, dim=0)
                    p.data.add_(normed, alpha=-lr)
                # Other parameters are not processed

class NormGradMomentumOptimizer(AbstractOptimizer):
    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.state = {}

    def step(self, closure=None):
        for group in self.param_groups:
            lr = group.get('lr', self.lr)
            momentum = group.get('momentum', self.momentum)
            weight_decay = group.get('weight_decay', self.weight_decay)
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                param_state = self.state.setdefault(p, {})
                if 'momentum_buffer' not in param_state:
                    buf = torch.zeros_like(grad)
                else:
                    buf = param_state['momentum_buffer']
                buf = momentum * buf + grad
                # Normalize momentum
                if buf.dim() == 2:
                    normed = F.normalize(buf, p=2, dim=0)
                elif buf.dim() == 1:
                    normed = F.normalize(buf, p=2, dim=0)
                else:
                    normed = buf
                # Add weight decay after normalization
                if weight_decay > 0:
                    normed = normed + weight_decay * p.data
                p.data.add_(normed, alpha=-lr)
                param_state['momentum_buffer'] = buf


class NormGradNesterovOptimizer(AbstractOptimizer):
    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.state = {}

    def step(self, closure=None):
        for group in self.param_groups:
            lr = group.get('lr', self.lr)
            momentum = group.get('momentum', self.momentum)
            weight_decay = group.get('weight_decay', self.weight_decay)
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                param_state = self.state.setdefault(p, {})
                
                # Add weight decay to gradient
                if weight_decay > 0:
                    grad = grad + weight_decay * p.data
                    
                if 'momentum_buffer' not in param_state:
                    buf = torch.zeros_like(grad)
                else:
                    buf = param_state['momentum_buffer']
                
                # Nesterov momentum: update momentum buffer first
                buf = momentum * buf + grad
                
                # Key of Nesterov: use "lookahead" gradient
                # Calculate lookahead gradient: grad + momentum * buf
                nesterov_grad = grad + momentum * buf
                
                # Normalize lookahead gradient
                if nesterov_grad.dim() == 2:
                    normed = F.normalize(nesterov_grad, p=2, dim=0)
                elif nesterov_grad.dim() == 1:
                    normed = F.normalize(nesterov_grad, p=2, dim=0)
                else:
                    normed = nesterov_grad
                
                # Parameter update
                p.data.add_(normed, alpha=-lr)
                param_state['momentum_buffer'] = buf




# CRNNPOptimizer: Column-row normalization RNNP
class CRNNPOptimizer(AbstractOptimizer):
    def __init__(self, params, lr=0.01, momentum=0.9, beta=0.95, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8):
        defaults = dict(lr=lr, momentum=momentum, beta=beta, weight_decay=weight_decay, betas=betas, eps=eps)
        super().__init__(params, defaults)
        self.lr = lr
        self.momentum = momentum
        self.beta = beta
        self.weight_decay = weight_decay
        self.betas = betas
        self.eps = eps
        self.state = {}

    def step(self, closure=None):
        for group in self.param_groups:
            lr = group.get('lr', self.lr)
            momentum = group.get('momentum', self.momentum)
            beta = group.get('beta', self.beta)
            weight_decay = group.get('weight_decay', self.weight_decay)
            betas = group.get('betas', self.betas)
            eps = group.get('eps', self.eps)
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                param_state = self.state.setdefault(p, {})
                if grad.dim() == 2:
                    # Momentum update
                    if 'momentum_buffer' not in param_state:
                        buf = torch.zeros_like(grad)
                    else:
                        buf = param_state['momentum_buffer']
                    buf.lerp_(grad, 1 - beta)
                    nesterov_buf = grad.lerp(buf, momentum)
                    # First calculate the column and row norms of the original momentum matrix
                    col_norm = nesterov_buf.norm(p=2, dim=0) # shape: [n_cols]
                    col_norm_sqrt = col_norm.sqrt().clamp_min(eps)
                    row_norm = nesterov_buf.norm(p=2, dim=1) # shape: [n_rows]
                    row_norm_sqrt = row_norm.sqrt().clamp_min(eps)
                    # First normalize by column, then by row
                    update = nesterov_buf / col_norm_sqrt.unsqueeze(0) / row_norm_sqrt.unsqueeze(1)
                    if weight_decay > 0:
                        p.data.mul_(1 - lr * weight_decay)
                    p.data.add_(update, alpha=-lr)
                    param_state['momentum_buffer'] = buf
                elif grad.dim() == 1 or grad.dim() == 0:
                    # Other parameters use Adam logic
                    if 'exp_avg' not in param_state:
                        param_state['exp_avg'] = torch.zeros_like(grad)
                        param_state['exp_avg_sq'] = torch.zeros_like(grad)
                        param_state['step'] = 0
                    exp_avg, exp_avg_sq = param_state['exp_avg'], param_state['exp_avg_sq']
                    param_state['step'] += 1
                    exp_avg.mul_(betas[0]).add_(grad, alpha=1-betas[0])
                    exp_avg_sq.mul_(betas[1]).addcmul_(grad, grad, value=1-betas[1])
                    bias_correction1 = 1 - betas[0] ** param_state['step']
                    bias_correction2 = 1 - betas[1] ** param_state['step']
                    step_size = lr * math.sqrt(bias_correction2) / bias_correction1
                    denom = exp_avg_sq.sqrt().add_(eps)
                    adam_update = exp_avg / denom
                    if weight_decay > 0:
                        p.data.mul_(1 - step_size * weight_decay)
                    p.data.add_(adam_update, alpha=-step_size)

# CRNNPhalfOptimizer: Like CRNNP but use norm (no sqrt) for row/col normalization, then divide by L2 norm of row norm vector
class CRNNPhalfOptimizer(AbstractOptimizer):
    def __init__(self, params, lr=0.01, momentum=0.9, beta=0.95, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8):
        defaults = dict(lr=lr, momentum=momentum, beta=beta, weight_decay=weight_decay, betas=betas, eps=eps)
        super().__init__(params, defaults)
        self.lr = lr
        self.momentum = momentum
        self.beta = beta
        self.weight_decay = weight_decay
        self.betas = betas
        self.eps = eps
        self.state = {}

    def step(self, closure=None):
        for group in self.param_groups:
            lr = group.get('lr', self.lr)
            momentum = group.get('momentum', self.momentum)
            beta = group.get('beta', self.beta)
            weight_decay = group.get('weight_decay', self.weight_decay)
            betas = group.get('betas', self.betas)
            eps = group.get('eps', self.eps)
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                param_state = self.state.setdefault(p, {})
                if grad.dim() == 2:
                    # Momentum update
                    if 'momentum_buffer' not in param_state:
                        buf = torch.zeros_like(grad)
                    else:
                        buf = param_state['momentum_buffer']
                    buf.lerp_(grad, 1 - beta)
                    nesterov_buf = grad.lerp(buf, momentum)
                    # Use norm (no sqrt) for col/row normalization
                    col_norm = nesterov_buf.norm(p=2, dim=0).clamp_min(eps)  # [n_cols]
                    row_norm = nesterov_buf.norm(p=2, dim=1).clamp_min(eps)  # [n_rows]
                    # Normalize by col, then by row
                    update = nesterov_buf / col_norm.unsqueeze(0) / row_norm.unsqueeze(1)
                    # Then divide by L2 norm of row_norm vector
                    row_norm_l2 = row_norm.norm(p=2).clamp_min(eps)
                    update = update * row_norm_l2
                    if weight_decay > 0:
                        p.data.mul_(1 - lr * weight_decay)
                    p.data.add_(update, alpha=-lr)
                    param_state['momentum_buffer'] = buf
                elif grad.dim() == 1 or grad.dim() == 0:
                    # Other parameters use Adam logic
                    if 'exp_avg' not in param_state:
                        param_state['exp_avg'] = torch.zeros_like(grad)
                        param_state['exp_avg_sq'] = torch.zeros_like(grad)
                        param_state['step'] = 0
                    exp_avg, exp_avg_sq = param_state['exp_avg'], param_state['exp_avg_sq']
                    param_state['step'] += 1
                    exp_avg.mul_(betas[0]).add_(grad, alpha=1-betas[0])
                    exp_avg_sq.mul_(betas[1]).addcmul_(grad, grad, value=1-betas[1])
                    bias_correction1 = 1 - betas[0] ** param_state['step']
                    bias_correction2 = 1 - betas[1] ** param_state['step']
                    step_size = lr * math.sqrt(bias_correction2) / bias_correction1
                    denom = exp_avg_sq.sqrt().add_(eps)
                    adam_update = exp_avg / denom
                    if weight_decay > 0:
                        p.data.mul_(1 - step_size * weight_decay)
                    p.data.add_(adam_update, alpha=-step_size)
class DSPOptimizer(AbstractOptimizer):
    def __init__(self, params, lr=0.01, weight_decay=0.0, eps=1e-8, accum_init=0):
        defaults = dict(lr=lr, weight_decay=weight_decay, eps=eps, accum_init=accum_init)
        super().__init__(params, defaults)
        self.lr = lr
        self.weight_decay = weight_decay
        self.eps = eps
        self.accum_init = accum_init
        self.state = {}

    def step(self, closure=None):
        for group in self.param_groups:
            lr = group.get('lr', self.lr)
            weight_decay = group.get('weight_decay', self.weight_decay)
            eps = group.get('eps', self.eps)
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                param_state = self.state.setdefault(p, {})
                if grad.dim() == 2:
                    # Calculate original column and row norms
                    col_norm = grad.norm(p=2, dim=0).clamp_min(eps)
                    row_norm = grad.norm(p=2, dim=1).clamp_min(eps)
                    # Accumulate original norms first
                    if 'col_accum' not in param_state:
                        param_state['col_accum'] = self.accum_init * torch.ones_like(col_norm)
                    if 'row_accum' not in param_state:
                        param_state['row_accum'] = self.accum_init * torch.ones_like(row_norm)
                    param_state['col_accum'] += col_norm
                    param_state['row_accum'] += row_norm
                    # Then take sqrt, then reciprocal
                    col_sqrt_accum = param_state['col_accum'].sqrt()
                    row_sqrt_accum = param_state['row_accum'].sqrt()
                    col_recip = 1.0 / col_sqrt_accum
                    row_recip = 1.0 / row_sqrt_accum
                    # Use reciprocal vectors to align and multiply with gradient
                    update = grad * col_recip.unsqueeze(0) * row_recip.unsqueeze(1)
                    if weight_decay > 0:
                        p.data.mul_(1 - lr * weight_decay)
                    p.data.add_(update, alpha=-lr)
                elif grad.dim() == 1 or grad.dim() == 0:
                    # 1D parameters use Adam logic
                    if 'exp_avg' not in param_state:
                        param_state['exp_avg'] = torch.zeros_like(grad)
                        param_state['exp_avg_sq'] = torch.zeros_like(grad)
                        param_state['step'] = 0
                    exp_avg, exp_avg_sq = param_state['exp_avg'], param_state['exp_avg_sq']
                    param_state['step'] += 1
                    beta1, beta2 = 0.9, 0.999
                    eps_adam = 1e-8
                    exp_avg.mul_(beta1).add_(grad, alpha=1-beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1-beta2)
                    bias_correction1 = 1 - beta1 ** param_state['step']
                    bias_correction2 = 1 - beta2 ** param_state['step']
                    step_size = lr * math.sqrt(bias_correction2) / bias_correction1
                    denom = exp_avg_sq.sqrt().add_(eps_adam)
                    adam_update = exp_avg / denom
                    if weight_decay > 0:
                        p.data.mul_(1 - step_size * weight_decay)
                    p.data.add_(adam_update, alpha=-step_size)

# CRNNP2Optimizer: Column-row normalization (directly use row/col norm, no sqrt)
class CRNNP2Optimizer(AbstractOptimizer):
    def __init__(self, params, lr=0.01, momentum=0.9, beta=0.95, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8):
        defaults = dict(lr=lr, momentum=momentum, beta=beta, weight_decay=weight_decay, betas=betas, eps=eps)
        super().__init__(params, defaults)
        self.lr = lr
        self.momentum = momentum
        self.beta = beta
        self.weight_decay = weight_decay
        self.betas = betas
        self.eps = eps
        self.state = {}

    def step(self, closure=None):
        for group in self.param_groups:
            lr = group.get('lr', self.lr)
            momentum = group.get('momentum', self.momentum)
            beta = group.get('beta', self.beta)
            weight_decay = group.get('weight_decay', self.weight_decay)
            betas = group.get('betas', self.betas)
            eps = group.get('eps', self.eps)
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                param_state = self.state.setdefault(p, {})
                if grad.dim() == 2:
                    if 'momentum_buffer' not in param_state:
                        buf = torch.zeros_like(grad)
                    else:
                        buf = param_state['momentum_buffer']
                    buf.lerp_(grad, 1 - beta)
                    nesterov_buf = grad.lerp(buf, momentum)
                    col_norm = nesterov_buf.norm(p=2, dim=0).clamp_min(eps)
                    row_norm = nesterov_buf.norm(p=2, dim=1).clamp_min(eps)
                    update = nesterov_buf / col_norm.unsqueeze(0) / row_norm.unsqueeze(1)
                    if weight_decay > 0:
                        p.data.mul_(1 - lr * weight_decay)
                    p.data.add_(update, alpha=-lr)
                    param_state['momentum_buffer'] = buf
                elif grad.dim() == 1 or grad.dim() == 0:
                    if 'exp_avg' not in param_state:
                        param_state['exp_avg'] = torch.zeros_like(grad)
                        param_state['exp_avg_sq'] = torch.zeros_like(grad)
                        param_state['step'] = 0
                    exp_avg, exp_avg_sq = param_state['exp_avg'], param_state['exp_avg_sq']
                    param_state['step'] += 1
                    exp_avg.mul_(betas[0]).add_(grad, alpha=1-betas[0])
                    exp_avg_sq.mul_(betas[1]).addcmul_(grad, grad, value=1-betas[1])
                    bias_correction1 = 1 - betas[0] ** param_state['step']
                    bias_correction2 = 1 - betas[1] ** param_state['step']
                    step_size = lr * math.sqrt(bias_correction2) / bias_correction1
                    denom = exp_avg_sq.sqrt().add_(eps)
                    adam_update = exp_avg / denom
                    if weight_decay > 0:
                        p.data.mul_(1 - step_size * weight_decay)
                    p.data.add_(adam_update, alpha=-step_size)

# CRNNPAOptimizer: Column-row normalization (directly use F.normalize)
class CRNNPAOptimizer(AbstractOptimizer):
    def __init__(self, params, lr=0.01, momentum=0.9, beta=0.95, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8):
        defaults = dict(lr=lr, momentum=momentum, beta=beta, weight_decay=weight_decay, betas=betas, eps=eps)
        super().__init__(params, defaults)
        self.lr = lr
        self.momentum = momentum
        self.beta = beta
        self.weight_decay = weight_decay
        self.betas = betas
        self.eps = eps
        self.state = {}

    def step(self, closure=None):
        for group in self.param_groups:
            lr = group.get('lr', self.lr)
            momentum = group.get('momentum', self.momentum)
            beta = group.get('beta', self.beta)
            weight_decay = group.get('weight_decay', self.weight_decay)
            betas = group.get('betas', self.betas)
            eps = group.get('eps', self.eps)
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                param_state = self.state.setdefault(p, {})
                if grad.dim() == 2:
                    if 'momentum_buffer' not in param_state:
                        buf = torch.zeros_like(grad)
                    else:
                        buf = param_state['momentum_buffer']
                    buf.lerp_(grad, 1 - beta)
                    nesterov_buf = grad.lerp(buf, momentum)
                    # 先对列归一化，再对行归一化
                    update = F.normalize(nesterov_buf, p=2, dim=0, eps=eps)
                    update = F.normalize(update, p=2, dim=1, eps=eps)
                    if weight_decay > 0:
                        p.data.mul_(1 - lr * weight_decay)
                    p.data.add_(update, alpha=-lr)
                    param_state['momentum_buffer'] = buf
                elif grad.dim() == 1 or grad.dim() == 0:
                    if 'exp_avg' not in param_state:
                        param_state['exp_avg'] = torch.zeros_like(grad)
                        param_state['exp_avg_sq'] = torch.zeros_like(grad)
                        param_state['step'] = 0
                    exp_avg, exp_avg_sq = param_state['exp_avg'], param_state['exp_avg_sq']
                    param_state['step'] += 1
                    exp_avg.mul_(betas[0]).add_(grad, alpha=1-betas[0])
                    exp_avg_sq.mul_(betas[1]).addcmul_(grad, grad, value=1-betas[1])
                    bias_correction1 = 1 - betas[0] ** param_state['step']
                    bias_correction2 = 1 - betas[1] ** param_state['step']
                    step_size = lr * math.sqrt(bias_correction2) / bias_correction1
                    denom = exp_avg_sq.sqrt().add_(eps)
                    adam_update = exp_avg / denom
                    if weight_decay > 0:
                        p.data.mul_(1 - step_size * weight_decay)
                    p.data.add_(adam_update, alpha=-step_size)

# CNNPOptimizer: Column normalization RNNP
class CNNPOptimizer(AbstractOptimizer):
    def __init__(self, params, lr=0.01, momentum=0.9, beta=0.95, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8):
        defaults = dict(lr=lr, momentum=momentum, beta=beta, weight_decay=weight_decay, betas=betas, eps=eps)
        super().__init__(params, defaults)
        self.lr = lr
        self.momentum = momentum
        self.beta = beta
        self.weight_decay = weight_decay
        self.betas = betas
        self.eps = eps
        self.state = {}

    def step(self, closure=None):
        for group in self.param_groups:
            lr = group.get('lr', self.lr)
            momentum = group.get('momentum', self.momentum)
            beta = group.get('beta', self.beta)
            weight_decay = group.get('weight_decay', self.weight_decay)
            betas = group.get('betas', self.betas)
            eps = group.get('eps', self.eps)
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                param_state = self.state.setdefault(p, {})
                if grad.dim() == 2:
                    if 'momentum_buffer' not in param_state:
                        buf = torch.zeros_like(grad)
                    else:
                        buf = param_state['momentum_buffer']
                    buf.lerp_(grad, 1 - beta)
                    nesterov_buf = grad.lerp(buf, momentum)
                    normed = F.normalize(nesterov_buf, p=2, dim=0)  # 列归一化
                    if weight_decay > 0:
                        p.data.mul_(1 - lr * weight_decay)
                    p.data.add_(normed, alpha=-lr)
                    param_state['momentum_buffer'] = buf
                elif grad.dim() == 1 or grad.dim() == 0:
                    if 'exp_avg' not in param_state:
                        param_state['exp_avg'] = torch.zeros_like(grad)
                        param_state['exp_avg_sq'] = torch.zeros_like(grad)
                        param_state['step'] = 0
                    exp_avg, exp_avg_sq = param_state['exp_avg'], param_state['exp_avg_sq']
                    param_state['step'] += 1
                    exp_avg.mul_(betas[0]).add_(grad, alpha=1-betas[0])
                    exp_avg_sq.mul_(betas[1]).addcmul_(grad, grad, value=1-betas[1])
                    bias_correction1 = 1 - betas[0] ** param_state['step']
                    bias_correction2 = 1 - betas[1] ** param_state['step']
                    step_size = lr * math.sqrt(bias_correction2) / bias_correction1
                    denom = exp_avg_sq.sqrt().add_(eps)
                    adam_update = exp_avg / denom
                    if weight_decay > 0:
                        p.data.mul_(1 - step_size * weight_decay)
                    p.data.add_(adam_update, alpha=-step_size)

