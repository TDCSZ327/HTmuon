import math
import torch
from torch import Tensor
from typing import List
from torch.optim import Optimizer


@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    # 始终在 bfloat16 中做这个迭代（保持你原来的设计）
    X = G.bfloat16()
    X /= (X.norm() + eps)  # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X


def soap_with_muon_gpt(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    exp_avgs_GG: List[Tensor],
    exp_avgs_P: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    state_steps: List[int],
    *,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    beta3: float,
    lr: float,
    weight_decay: float,
    eps: float,
    maximize: bool,
    ratio: float,
):

    for i, param in enumerate(params):

        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        exp_avg_gg = exp_avgs_GG[i]
        exp_avg_p = exp_avgs_P[i]

        step = state_steps[i]

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
        bias_correction3 = 1 - beta3 ** step  # 预留，保持不变

        # 标准一阶动量
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

        if len(param.size()) == 2 and param.size(0) <= 10000:
            # 低秩 GNG 逻辑
            param_dtype = param.dtype

            if step == 1:
                # 用 float32 做 SVD，避免 bf16 不支持
                G32 = grad.to(torch.float32)               # [m, n]
                W32 = torch.matmul(G32, G32.T)             # [m, m]
                U32, _, _ = torch.linalg.svd(W32, full_matrices=False)

                rank = exp_avg_gg.size(0)
                P32 = U32[:, :rank]                        # [m, rank] float32

                # 写回 P（保持 state 的 dtype，比如 bfloat16）
                exp_avg_p.copy_(P32.to(param_dtype))

                # 初始化 GG，仍然在 float32 里算，再 cast 回去
                GG32 = torch.matmul(torch.matmul(P32.T, W32), P32) * (1 - beta3)
                exp_avg_gg.copy_(GG32.to(param_dtype))

            else:
                # 后续迭代，同样全部在 float32 里做，再写回
                P32 = exp_avg_p.to(torch.float32)          # [m, r]
                GG32 = exp_avg_gg.to(torch.float32)        # [r, r]
                G32 = grad.to(torch.float32)               # [m, n]

                t32 = P32.T.clone()                        # [r, m]

                P32 = beta3 * torch.matmul(P32, GG32) + \
                      (1 - beta3) * torch.matmul(G32, torch.matmul(G32.T, P32))
                P32, _ = torch.linalg.qr(P32, mode='reduced')  # [m, r]

                t32 = torch.matmul(t32, P32)               # [r, r]

                GG32 = beta3 * torch.matmul(t32.T, torch.matmul(GG32, t32)) + \
                        (1 - beta3) * torch.matmul(
                            torch.matmul(G32.T, P32).T,
                            torch.matmul(G32.T, P32)
                        )

                # 写回 state，保持 dtype 一致
                exp_avg_p.copy_(P32.to(param_dtype))
                exp_avg_gg.copy_(GG32.to(param_dtype))

            # 下面保持你原来的逻辑，只是保证 dtype 一致
            scale = (grad.size(0) * grad.size(1)) ** 0.5
            low_rank_grad = torch.matmul(exp_avg_p.T, grad)
            exp_avg_sq.mul_(beta2).addcmul_(low_rank_grad, low_rank_grad.conj(), value=1 - beta2)

            t = torch.matmul(exp_avg_p.T, exp_avg)
            step_size = lr / bias_correction1
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
            t1 = step_size * torch.matmul(exp_avg_p, t / denom)

            # 正交补方向 + zeropower
            t = exp_avg - torch.matmul(exp_avg_p, t)
            if t.size(1) == 3 * t.size(0):
                t = torch.cat(
                    [zeropower_via_newtonschulz5(g1, steps=5) for g1 in t.split(t.size(0), dim=1)],
                    dim=1,
                )
            else:
                t = zeropower_via_newtonschulz5(t, steps=5)

            t = t / (t.norm() + eps)
            t1.add_(t, alpha=scale * ratio * lr)
            param.add_(t1 / (t1.norm() + eps), alpha=-scale * ratio * lr)

        else:
            # 普通 AdamW 分支保持不变
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
                # Use the max. for normalizing running avg. of gradient
                denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(eps)
            else:
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

            step_size = lr / bias_correction1
            param.addcdiv_(exp_avg, denom, value=-step_size)


class COSMOS_for_gpt(Optimizer):

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.96, 0.96),
        eps=1e-8,
        lr_ratio=0.1,
        rank=64,
        weight_decay=0,
        amsgrad=False,
        *,
        maximize: bool = False,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError("Invalid beta parameter at index 2: {}".format(betas[2]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            maximize=maximize,
        )
        super(COSMOS_for_gpt, self).__init__(params, defaults)
        self.lr_ratio = lr_ratio
        self.rank = rank

    def __setstate__(self, state):
        super(COSMOS_for_gpt, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)
            group.setdefault("maximize", False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            exp_avgs_GG = []
            exp_avgs_P = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2, beta3 = group["betas"]

            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError("COSMOS does not support sparse gradients.")
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state["step"] = 0
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                        if len(p.size()) == 2 and p.size(0) <= 10000:
                            state["exp_avg_GG"] = torch.zeros(
                                self.rank, self.rank, dtype=p.data.dtype, device=p.data.device
                            )
                            state["exp_avg_P"] = torch.zeros(
                                p.size(0), self.rank, dtype=p.data.dtype, device=p.data.device
                            )
                            state["exp_avg_sq"] = torch.zeros(
                                self.rank, p.size(1), dtype=p.data.dtype, device=p.data.device
                            )
                        else:
                            state["exp_avg_GG"] = torch.zeros(0)
                            state["exp_avg_P"] = torch.zeros(0)
                            state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                        if group["amsgrad"]:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state["max_exp_avg_sq"] = torch.zeros_like(
                                p, memory_format=torch.preserve_format
                            )

                    exp_avgs.append(state["exp_avg"])
                    exp_avg_sqs.append(state["exp_avg_sq"])
                    exp_avgs_GG.append(state["exp_avg_GG"])
                    exp_avgs_P.append(state["exp_avg_P"])

                    if group["amsgrad"]:
                        max_exp_avg_sqs.append(state["max_exp_avg_sq"])

                    # update the steps for each param group update
                    state["step"] += 1
                    # record the step after step update
                    state_steps.append(state["step"])

            soap_with_muon_gpt(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                exp_avgs_GG,
                exp_avgs_P,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=group["amsgrad"],
                beta1=beta1,
                beta2=beta2,
                beta3=beta3,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=group["maximize"],
                ratio=self.lr_ratio,
            )

        return loss


def soap_with_muon_llama(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    exp_avgs_GG: List[Tensor],
    exp_avgs_P: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    state_steps: List[int],
    *,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    beta3: float,
    lr: float,
    weight_decay: float,
    eps: float,
    maximize: bool,
    ratio: float,
):

    for i, param in enumerate(params):

        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        exp_avg_gg = exp_avgs_GG[i]
        exp_avg_p = exp_avgs_P[i]

        step = state_steps[i]

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
        bias_correction3 = 1 - beta3 ** step  # 预留

        if len(param.size()) == 2 and param.size(0) <= 10000:
            param_dtype = param.dtype

            # 一阶动量先更新
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

            if step == 1:
                # 用 float32 做 SVD，避免 bf16 不支持
                G32 = grad.to(torch.float32)         # [m, n]
                W32 = torch.matmul(G32.T, G32)       # [n, n]
                U32, _, _ = torch.linalg.svd(W32, full_matrices=False)

                rank = exp_avg_gg.size(0)
                P32 = U32[:, :rank]                 # [n, rank]

                # 写回 P（state），注意 llama 里 P 是 (p.size(1), rank)
                exp_avg_p.copy_(P32.to(param_dtype))

                # 用 float32 计算 GG = (P^T G^T)(G P) * (1 - beta3)
                GG32 = torch.matmul(
                    torch.matmul(P32.T, G32.T),
                    torch.matmul(G32, P32),
                ) * (1 - beta3)
                exp_avg_gg.copy_(GG32.to(param_dtype))

            else:
                # 后续迭代，全程 float32 做，再写回
                P32 = exp_avg_p.to(torch.float32)    # [n, r]
                GG32 = exp_avg_gg.to(torch.float32)  # [r, r]
                G32 = grad.to(torch.float32)         # [m, n]

                t32 = P32.T.clone()                  # [r, n]

                P32 = beta3 * torch.matmul(P32, GG32) + \
                      (1 - beta3) * torch.matmul(G32.T, torch.matmul(G32, P32))
                P32, _ = torch.linalg.qr(P32, mode="reduced")  # [n, r]

                t32 = torch.matmul(t32, P32)         # [r, r]

                GG32 = beta3 * torch.matmul(t32.T, torch.matmul(GG32, t32)) + \
                        (1 - beta3) * torch.matmul(
                            torch.matmul(G32, P32).T,
                            torch.matmul(G32, P32),
                        )

                exp_avg_p.copy_(P32.to(param_dtype))
                exp_avg_gg.copy_(GG32.to(param_dtype))

            scale = (grad.size(0) * grad.size(1)) ** 0.5
            low_rank_grad = torch.matmul(grad, exp_avg_p)
            exp_avg_sq.mul_(beta2).addcmul_(low_rank_grad, low_rank_grad.conj(), value=1 - beta2)

            t = torch.matmul(exp_avg, exp_avg_p)
            step_size = lr / bias_correction1
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
            t1 = t / denom * step_size
            t1 = torch.matmul(t1, exp_avg_p.T)

            t = exp_avg - torch.matmul(t, exp_avg_p.T)
            t = zeropower_via_newtonschulz5(t, steps=5)
            t = t / (t.norm() + eps)
            t1.add_(t, alpha=scale * ratio * lr)
            param.add_(t1 / (t1.norm() + eps), alpha=-scale * ratio * lr)

        else:
            # 非 2D / 大矩阵，普通 AdamW 分支
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)

            if amsgrad:
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
                denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(eps)
            else:
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

            step_size = lr / bias_correction1
            param.addcdiv_(exp_avg, denom, value=-step_size)


class COSMOS_for_llama(Optimizer):

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.98, 0.98),
        eps=1e-8,
        lr_ratio=0.1,
        rank=64,
        weight_decay=0,
        amsgrad=False,
        *,
        maximize: bool = False,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError("Invalid beta parameter at index 2: {}".format(betas[2]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            maximize=maximize,
        )
        super(COSMOS_for_llama, self).__init__(params, defaults)
        self.lr_ratio = lr_ratio
        self.rank = rank

    def __setstate__(self, state):
        super(COSMOS_for_llama, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)
            group.setdefault("maximize", False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            exp_avgs_GG = []
            exp_avgs_P = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2, beta3 = group["betas"]

            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError("COSMOS does not support sparse gradients.")
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state["step"] = 0
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                        if len(p.size()) == 2 and p.size(0) <= 10000:
                            state["exp_avg_GG"] = torch.zeros(
                                self.rank, self.rank, dtype=p.data.dtype, device=p.data.device
                            )
                            state["exp_avg_P"] = torch.zeros(
                                p.size(1), self.rank, dtype=p.data.dtype, device=p.data.device
                            )
                            state["exp_avg_sq"] = torch.zeros(
                                p.size(0), self.rank, dtype=p.data.dtype, device=p.data.device
                            )
                        else:
                            state["exp_avg_GG"] = torch.zeros(0)
                            state["exp_avg_P"] = torch.zeros(0)
                            state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                        if group["amsgrad"]:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state["max_exp_avg_sq"] = torch.zeros_like(
                                p, memory_format=torch.preserve_format
                            )

                    exp_avgs.append(state["exp_avg"])
                    exp_avg_sqs.append(state["exp_avg_sq"])
                    exp_avgs_GG.append(state["exp_avg_GG"])
                    exp_avgs_P.append(state["exp_avg_P"])

                    if group["amsgrad"]:
                        max_exp_avg_sqs.append(state["max_exp_avg_sq"])

                    state["step"] += 1
                    state_steps.append(state["step"])

            soap_with_muon_llama(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                exp_avgs_GG,
                exp_avgs_P,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=group["amsgrad"],
                beta1=beta1,
                beta2=beta2,
                beta3=beta3,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=group["maximize"],
                ratio=self.lr_ratio,
            )

        return loss
