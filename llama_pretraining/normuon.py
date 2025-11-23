import torch
import torch.distributed as dist
import math
# copied from https://github.com/KellerJordan/Muon/blob/master/muon.py
def zeropower_via_newtonschulz5(G, steps=5):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X



def zeropower_via_newtonschulz_exact(G: torch.Tensor,
                                     steps: int = 10,
                                     eps: float = 1e-6) -> torch.Tensor:
    """
    Newton–Schulz iteration to compute the polar factor UV^T of G
    (i.e. the 'zeroth power' / orthogonalization, exact version).

    This is the classical NS polar iteration:
        X_{k+1} = 0.5 * X_k (3I - X_k^T X_k)

    Args:
        G: (..., m, n) input matrix (can be batched)
        steps: number of NS iterations; 5–10 usually enough in fp32
        eps: small constant to avoid division by zero

    Returns:
        X: (..., m, n), approx UV^T with orthonormal columns (if m >= n)
    """
    assert G.ndim >= 2

    # 用高一点的精度算
    X = G.float()

    # 如果是宽矩阵，先转成 tall，再转回去
    transposed = False
    if X.size(-2) < X.size(-1):
        X = X.mT
        transposed = True

    # 预缩放：保证谱半径 <= 1（Fro 范数是谱范数的上界）
    frob = X.norm(dim=(-2, -1), keepdim=True)
    X = X / (frob + eps)

    n = X.size(-1)
    I = torch.eye(n, device=X.device, dtype=X.dtype)

    # Newton–Schulz 迭代：收敛到 UV^T
    for _ in range(steps):
        XtX = X.mT @ X                    # (..., n, n)
        X = 0.5 * X @ (3.0 * I - XtX)     # (..., m, n)

    if transposed:
        X = X.mT

    # cast 回原来的 dtype（比如 bfloat16）
    return X.to(G.dtype)


def gram_root_1_16_via_newtonschulz(G, power, ns_steps: int = 5, eps: float = 1e-6):
    """
    Approximate (G^T G)^{1/16} using 4x Newton–Schulz matrix square root iterations.

    G: (..., m, n)
    return: (..., n, n) SPD matrix approximating (G^T G)^{1/16}
    """
    assert G.ndim >= 2

    # 工作精度建议至少 fp32，再根据需要 cast 回去
    G_work = G.float()

    # Gram matrix A = G^T G, shape (..., n, n)
    A = G_work.transpose(-2, -1) @ G_work

    # 轻微正则，防止奇异
    n = A.size(-1)
    I_base = torch.eye(n, device=A.device, dtype=A.dtype)
    I = I_base.expand(A.shape)  # broadcast 到 batch 维度
    A = A + eps * I

    S = A  # 当前的 SPD 矩阵，开始是 A，本层结束后会变成 A^(1/2), 再给下一层
    val = math.log(power, 0.5) + 1   # 理论上就是 4.0
    val = int(round(val))

    for _ in range(val):  # 4 次 sqrt -> 1/16 次方
        # 1) 缩放，保证谱半径适中
        alpha = S.norm(dim=(-2, -1), keepdim=True)  # Fro 范数
        alpha = alpha.clamp_min(eps)                # 防止除零
        S_tilde = S / alpha

        # 2) Newton–Schulz for matrix square root of S_tilde
        Y = S_tilde.clone()
        Z = I_base.expand(S_tilde.shape).clone()    # 每层新的 I，避免梯度奇怪广播

        for _ in range(ns_steps):
            T = 3.0 * I - Z @ Y          # 3I - Z Y
            Y = 0.5 * (Y @ T)            # Y_{k+1}
            Z = 0.5 * (T @ Z)            # Z_{k+1}

        # 3) 还原缩放
        S = (alpha.sqrt()) * Y          # 现在 S ≈ 原 S 的 1/2 次方

        # （可选）再加一次 tiny 正则，保证 SPD
        S = 0.5 * (S + S.transpose(-2, -1))         # symmetrize
        S = S + eps * I

    # 此时 S ≈ (G^T G)^{1/16}
    # 如需 bfloat16 可以在最后转换
    return S.to(G.dtype)

# def normuon_update(grad, momentum, second_momentum, beta=0.95, beta2=0.95, ns_steps=5, nesterov=True):
#     momentum.lerp_(grad, 1 - beta)
#     update = grad.lerp_(momentum, beta) if nesterov else momentum
#     original_shape = None
#     if update.ndim == 4:  # for the case of conv filters
#         original_shape = update.shape
#         update = update.reshape(update.size(0), -1)
#     update = zeropower_via_newtonschulz5(update, steps=ns_steps).float()
#     if original_shape is not None:
#         update = update.reshape(original_shape)
#     ################ NorMuon added ###################
#     vnorm = update.norm(dim=(-2,-1), keepdim=True)
#     v_mean = torch.mean(update * update, dim=-1, keepdim=True)
#     second_momentum.lerp_(v_mean, 1 - beta2)
#     step_size = 1 / second_momentum.sqrt().add_(1e-10)
#     update.mul_(step_size)
#     vnorm_new = update.norm(dim=(-2,-1), keepdim=True)
#     update.mul_(vnorm / (vnorm_new.add_(1e-10))) # This scaling keep the update norm the same as pre-normalization
#     ##################################################
#     update *= max(1, grad.size(-2) / grad.size(-1))**0.5
#     return update


def taylor_sqrt_poly(S, I_base, degree: int = 4, eps: float = 1e-6):
    """
    用 sqrt(1 + u) 在 u=0 处的泰勒展开来近似 sqrt(S) 的一层（矩阵）开方。
    S: (..., n, n) SPD
    I_base: (n, n) 的单位矩阵（不含 batch 维度）
    degree: 使用到的泰勒展开阶数（1~4）
    """
    # 1) broadcast 单位阵
    I = I_base.expand(S.shape)

    # 2) 缩放，尽量把谱压到 ~1 附近，方便在 I 附近展开
    alpha = S.norm(dim=(-2, -1), keepdim=True)   # Fro 范数
    alpha = alpha.clamp_min(eps)
    S_tilde = S / alpha                           # 期望特征值在 (0, 1] 这一带

    # 3) 在 I 附近做泰勒展开：sqrt(S_tilde) ≈ P(S_tilde)
    #    设 S_tilde = I + X， X = S_tilde - I
    #    sqrt(1 + u) = 1
    #                 + (1/2) u
    #                 - (1/8) u^2
    #                 + (1/16) u^3
    #                 - (5/128) u^4 + ...
    X = S_tilde - I

    # 常数项 + 一次项
    Y = I + 0.5 * X

    if degree >= 2:
        X2 = X @ X
        Y = Y - 0.125 * X2      # -1/8

    if degree >= 3:
        X3 = X2 @ X
        Y = Y + 0.0625 * X3     # 1/16

    if degree >= 4:
        X4 = X3 @ X
        Y = Y - 5.0/128.0 * X4  # -5/128

    # 4) 还原缩放：sqrt(S) ≈ sqrt(alpha) * sqrt(S_tilde)
    S_sqrt = (alpha.sqrt()) * Y

    return S_sqrt


def gram_root_1_16_via_taylor(G, power: float = 0.125, taylor_degree: int = 4,
                              eps: float = 1e-6):
    """
    用 4 次“泰勒多项式矩阵开方”近似 (G^T G)^{1/16}。

    G: (..., m, n)
    return: (..., n, n) 近似 (G^T G)^{1/16} 的 SPD 矩阵
    """
    assert G.ndim >= 2

    # 建议用 fp32 计算
    G_work = G.float()

    # Gram matrix A = G^T G, (..., n, n)
    A = G_work.transpose(-2, -1) @ G_work

    # 轻微正则防止奇异
    n = A.size(-1)
    I_base = torch.eye(n, device=A.device, dtype=A.dtype)
    I = I_base.expand(A.shape)
    A = A + eps * I

    # 初始 S = A
    S = A

    # 根据 power 来决定需要几次 sqrt，power = (1/2)^k => k = log_{1/2}(power)
    # 对于 1/16，power = 1/16 => k = 4
    steps = math.log(power, 0.5) + 1   # 理论上就是 4.0
    steps = int(round(steps))                      # 保险

    for i in range(steps):
        # 用泰勒多项式做一次矩阵开方：S -> S^{1/2}
        S = taylor_sqrt_poly(S, I_base, degree=taylor_degree, eps=eps)

        # 保持 SPD 性质：对称化 + 微正则
        S = 0.5 * (S + S.transpose(-2, -1))
        S = S + eps * I
        # 若想看每层结果可以打开这一行：
        # print(f"Iteration {i+1}: approx A^(1/{2**(i+1)})")

    # 此时 S ≈ (G^T G)^{power}，默认 power=1/16
    return S.to(G.dtype)



def msign_generalized_ht(g: torch.Tensor, p: float = 0.25) -> torch.Tensor:
    orig_dtype = g.dtype
    U32, S32, Vh32 = torch.linalg.svd(g.to(torch.float32), full_matrices=False)
    k = S32.shape[-1]
    idx = torch.arange(1, k + 1, device=S32.device, dtype=S32.dtype)
    d = idx.pow(-p)
    out32 = (U32 * d.view(*([1] * (U32.ndim - 2)), 1, k)) @ Vh32
    return out32.to(orig_dtype)

def msign_generalized_ht_v2(g: torch.Tensor, p: float = 0.25) -> torch.Tensor:
    orig_dtype = g.dtype
    U32, S32, Vh32 = torch.linalg.svd(g.to(torch.float32), full_matrices=False)
    k = S32.shape[-1]

    # 对 S32 每一维做 p 次方
    Sp32 = S32.pow(p)

    # 把处理后的奇异值当作列缩放因子乘回 U
    Sp32_view = Sp32.view(*([1] * (U32.ndim - 2)), 1, k)
    out32 = (U32 * Sp32_view) @ Vh32

    return out32.to(orig_dtype)




def muon_generalized_ht_update(grad, momentum, power=0.25, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4: # for the case of conv filters
        update = update.view(len(update), -1)
    update = msign_generalized_ht(update,power)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update

@torch.no_grad()
def normuon_generalized_ht_update(grad, momentum, second_momentum,power=0.25,
                   beta=0.95, beta2=0.95, ns_steps=5, nesterov=True):
    # 统一 dtype/device（继承 state 的 bf16）
    dtype, device = momentum.dtype, momentum.device
    g = grad if grad.dtype == dtype else grad.to(dtype)

    # 一阶动量
    momentum.lerp_(g, 1.0 - beta)

    # Nesterov：不要在 grad 上 in-place
    update = torch.lerp(g, momentum, beta) if nesterov else momentum

    # conv 权重摊平
    original_shape = None
    if update.ndim == 4:
        original_shape = update.shape
        update = update.reshape(update.size(0), -1)

    # 重要：保持 bf16，禁止 .float()
    update = msign_generalized_ht(update,power)
    if update.dtype != dtype:
        update = update.to(dtype)

    if original_shape is not None:
        update = update.reshape(original_shape)

    # ---------- NorMuon ----------
    vnorm = update.norm(dim=(-2, -1), keepdim=True)

    # v_mean 与 second_momentum 同 dtype
    try:
        v_mean = (update * update).mean(dim=-1, keepdim=True, dtype=dtype)
    except TypeError:
        v_mean = (update * update).mean(dim=-1, keepdim=True)
        if v_mean.dtype != dtype:
            v_mean = v_mean.to(dtype)

    second_momentum.lerp_(v_mean, 1.0 - beta2)

    # step_size 与 eps 也用同 dtype/device
    eps_t = torch.tensor(1e-10, dtype=dtype, device=device)
    step_size = torch.rsqrt(second_momentum + eps_t)
    update.mul_(step_size)

    vnorm_new = update.norm(dim=(-2, -1), keepdim=True)
    update.mul_(vnorm / (vnorm_new + eps_t))  # 保持归一化前的范数

    # 尺度修正（标量也对齐 dtype）
    scale = max(1.0, (grad.size(-2) / grad.size(-1)) ** 0.5)
    update.mul_(torch.tensor(scale, dtype=dtype, device=device))

    return update

@torch.no_grad()
def normuon_generalized_ht_update_v2(grad, momentum, second_momentum,power=0.25,
                   beta=0.95, beta2=0.95, ns_steps=5, nesterov=True):
    # 统一 dtype/device（继承 state 的 bf16）
    dtype, device = momentum.dtype, momentum.device
    g = grad if grad.dtype == dtype else grad.to(dtype)

    # 一阶动量
    momentum.lerp_(g, 1.0 - beta)

    # Nesterov：不要在 grad 上 in-place
    update = torch.lerp(g, momentum, beta) if nesterov else momentum

    # conv 权重摊平
    original_shape = None
    if update.ndim == 4:
        original_shape = update.shape
        update = update.reshape(update.size(0), -1)

    # 重要：保持 bf16，禁止 .float()
    update = msign_generalized_ht_v2(update,power)
    if update.dtype != dtype:
        update = update.to(dtype)

    if original_shape is not None:
        update = update.reshape(original_shape)

    # ---------- NorMuon ----------
    vnorm = update.norm(dim=(-2, -1), keepdim=True)

    # v_mean 与 second_momentum 同 dtype
    try:
        v_mean = (update * update).mean(dim=-1, keepdim=True, dtype=dtype)
    except TypeError:
        v_mean = (update * update).mean(dim=-1, keepdim=True)
        if v_mean.dtype != dtype:
            v_mean = v_mean.to(dtype)

    second_momentum.lerp_(v_mean, 1.0 - beta2)

    # step_size 与 eps 也用同 dtype/device
    eps_t = torch.tensor(1e-10, dtype=dtype, device=device)
    step_size = torch.rsqrt(second_momentum + eps_t)
    update.mul_(step_size)

    vnorm_new = update.norm(dim=(-2, -1), keepdim=True)
    update.mul_(vnorm / (vnorm_new + eps_t))  # 保持归一化前的范数

    # 尺度修正（标量也对齐 dtype）
    scale = max(1.0, (grad.size(-2) / grad.size(-1)) ** 0.5)
    update.mul_(torch.tensor(scale, dtype=dtype, device=device))

    return update


@torch.no_grad()
def normuon_generalized_ht_update_v2_acc(grad, momentum, second_momentum,power=0.25,
                   beta=0.95, beta2=0.95, ns_steps=5, nesterov=True):
    # 统一 dtype/device（继承 state 的 bf16）
    dtype, device = momentum.dtype, momentum.device
    g = grad if grad.dtype == dtype else grad.to(dtype)

    # 一阶动量
    momentum.lerp_(g, 1.0 - beta)

    # Nesterov：不要在 grad 上 in-place
    update = torch.lerp(g, momentum, beta) if nesterov else momentum

    # conv 权重摊平
    original_shape = None
    if update.ndim == 4:
        original_shape = update.shape
        update = update.reshape(update.size(0), -1)

    # 重要：保持 bf16，禁止 .float()
    update_1 = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update_2 = gram_root_1_16_via_newtonschulz(update, power,ns_steps=ns_steps)
    update = update_1 @ update_2
    if update.dtype != dtype:
        update = update.to(dtype)

    if original_shape is not None:
        update = update.reshape(original_shape)

    # ---------- NorMuon ----------
    vnorm = update.norm(dim=(-2, -1), keepdim=True)

    # v_mean 与 second_momentum 同 dtype
    try:
        v_mean = (update * update).mean(dim=-1, keepdim=True, dtype=dtype)
    except TypeError:
        v_mean = (update * update).mean(dim=-1, keepdim=True)
        if v_mean.dtype != dtype:
            v_mean = v_mean.to(dtype)

    second_momentum.lerp_(v_mean, 1.0 - beta2)

    # step_size 与 eps 也用同 dtype/device
    eps_t = torch.tensor(1e-10, dtype=dtype, device=device)
    step_size = torch.rsqrt(second_momentum + eps_t)
    update.mul_(step_size)

    vnorm_new = update.norm(dim=(-2, -1), keepdim=True)
    update.mul_(vnorm / (vnorm_new + eps_t))  # 保持归一化前的范数

    # 尺度修正（标量也对齐 dtype）
    scale = max(1.0, (grad.size(-2) / grad.size(-1)) ** 0.5)
    update.mul_(torch.tensor(scale, dtype=dtype, device=device))

    return update


@torch.no_grad()
def normuon_generalized_ht_update_v2_accv2(grad, momentum, second_momentum,power=0.25,
                   beta=0.95, beta2=0.95, ns_steps=10, nesterov=True):
    # 统一 dtype/device（继承 state 的 bf16）
    dtype, device = momentum.dtype, momentum.device
    g = grad if grad.dtype == dtype else grad.to(dtype)

    # 一阶动量
    momentum.lerp_(g, 1.0 - beta)

    # Nesterov：不要在 grad 上 in-place
    update = torch.lerp(g, momentum, beta) if nesterov else momentum

    # conv 权重摊平
    original_shape = None
    if update.ndim == 4:
        original_shape = update.shape
        update = update.reshape(update.size(0), -1)

    # 重要：保持 bf16，禁止 .float()
    update_1 = zeropower_via_newtonschulz_exact(update, steps=ns_steps)
    update_2 = gram_root_1_16_via_newtonschulz(update, power,ns_steps=ns_steps)
    update = update_1 @ update_2
    if update.dtype != dtype:
        update = update.to(dtype)

    if original_shape is not None:
        update = update.reshape(original_shape)

    # ---------- NorMuon ----------
    vnorm = update.norm(dim=(-2, -1), keepdim=True)

    # v_mean 与 second_momentum 同 dtype
    try:
        v_mean = (update * update).mean(dim=-1, keepdim=True, dtype=dtype)
    except TypeError:
        v_mean = (update * update).mean(dim=-1, keepdim=True)
        if v_mean.dtype != dtype:
            v_mean = v_mean.to(dtype)

    second_momentum.lerp_(v_mean, 1.0 - beta2)

    # step_size 与 eps 也用同 dtype/device
    eps_t = torch.tensor(1e-10, dtype=dtype, device=device)
    step_size = torch.rsqrt(second_momentum + eps_t)
    update.mul_(step_size)

    vnorm_new = update.norm(dim=(-2, -1), keepdim=True)
    update.mul_(vnorm / (vnorm_new + eps_t))  # 保持归一化前的范数

    # 尺度修正（标量也对齐 dtype）
    scale = max(1.0, (grad.size(-2) / grad.size(-1)) ** 0.5)
    update.mul_(torch.tensor(scale, dtype=dtype, device=device))

    return update




@torch.no_grad()
def normuon_generalized_ht_update_v2_accv3(grad, momentum, second_momentum,power,
                   beta=0.95, beta2=0.95, ns_steps=5, nesterov=True):
    # 统一 dtype/device（继承 state 的 bf16）
    dtype, device = momentum.dtype, momentum.device
    g = grad if grad.dtype == dtype else grad.to(dtype)

    # 一阶动量
    momentum.lerp_(g, 1.0 - beta)

    # Nesterov：不要在 grad 上 in-place
    update = torch.lerp(g, momentum, beta) if nesterov else momentum

    # conv 权重摊平
    original_shape = None
    if update.ndim == 4:
        original_shape = update.shape
        update = update.reshape(update.size(0), -1)

    # 重要：保持 bf16，禁止 .float()
    update_1 = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update_2 = gram_root_1_16_via_taylor(update, power)
    update = update_1 @ update_2
    if update.dtype != dtype:
        update = update.to(dtype)

    if original_shape is not None:
        update = update.reshape(original_shape)

    # ---------- NorMuon ----------
    vnorm = update.norm(dim=(-2, -1), keepdim=True)

    # v_mean 与 second_momentum 同 dtype
    try:
        v_mean = (update * update).mean(dim=-1, keepdim=True, dtype=dtype)
    except TypeError:
        v_mean = (update * update).mean(dim=-1, keepdim=True)
        if v_mean.dtype != dtype:
            v_mean = v_mean.to(dtype)

    second_momentum.lerp_(v_mean, 1.0 - beta2)

    # step_size 与 eps 也用同 dtype/device
    eps_t = torch.tensor(1e-10, dtype=dtype, device=device)
    step_size = torch.rsqrt(second_momentum + eps_t)
    update.mul_(step_size)

    vnorm_new = update.norm(dim=(-2, -1), keepdim=True)
    update.mul_(vnorm / (vnorm_new + eps_t))  # 保持归一化前的范数

    # 尺度修正（标量也对齐 dtype）
    scale = max(1.0, (grad.size(-2) / grad.size(-1)) ** 0.5)
    update.mul_(torch.tensor(scale, dtype=dtype, device=device))

    return update


@torch.no_grad()
def normuon_generalized_ht_update_rms(grad, momentum, second_momentum,power=0.25,
                   beta=0.95, beta2=0.95, ns_steps=5, nesterov=True):
    # 统一 dtype/device（继承 state 的 bf16）
    dtype, device = momentum.dtype, momentum.device
    g = grad if grad.dtype == dtype else grad.to(dtype)

    # 一阶动量
    momentum.lerp_(g, 1.0 - beta)

    # Nesterov：不要在 grad 上 in-place
    update = torch.lerp(g, momentum, beta) if nesterov else momentum

    # conv 权重摊平
    original_shape = None
    if update.ndim == 4:
        original_shape = update.shape
        update = update.reshape(update.size(0), -1)

    # 重要：保持 bf16，禁止 .float()
    update = msign_generalized_ht(update,power)
    if update.dtype != dtype:
        update = update.to(dtype)

    if original_shape is not None:
        update = update.reshape(original_shape)

    # ---------- NorMuon ----------
    vnorm = update.norm(dim=(-2, -1), keepdim=True)

    # v_mean 与 second_momentum 同 dtype
    try:
        v_mean = (update * update).mean(dim=-1, keepdim=True, dtype=dtype)
    except TypeError:
        v_mean = (update * update).mean(dim=-1, keepdim=True)
        if v_mean.dtype != dtype:
            v_mean = v_mean.to(dtype)

    second_momentum.lerp_(v_mean, 1.0 - beta2)

    # step_size 与 eps 也用同 dtype/device
    eps_t = torch.tensor(1e-10, dtype=dtype, device=device)
    step_size = torch.rsqrt(second_momentum + eps_t)
    update.mul_(step_size)

    vnorm_new = update.norm(dim=(-2, -1), keepdim=True)
    update.mul_(vnorm / (vnorm_new + eps_t))  # 保持归一化前的范数

    # 尺度修正（标量也对齐 dtype）
    #scale = max(1.0, (grad.size(-2) / grad.size(-1)) ** 0.5)
    scale = 0.2 * (grad.size(-2) * grad.size(-1))**0.5 / (update.norm() + eps_t)
    #update.mul_(torch.tensor(scale, dtype=dtype, device=device))
    update*=scale

    return update




@torch.no_grad()
def normuon_update(grad, momentum, second_momentum,
                   beta=0.95, beta2=0.95, ns_steps=5, nesterov=True):
    # 统一 dtype/device（继承 state 的 bf16）
    dtype, device = momentum.dtype, momentum.device
    g = grad if grad.dtype == dtype else grad.to(dtype)

    # 一阶动量
    momentum.lerp_(g, 1.0 - beta)

    # Nesterov：不要在 grad 上 in-place
    update = torch.lerp(g, momentum, beta) if nesterov else momentum

    # conv 权重摊平
    original_shape = None
    if update.ndim == 4:
        original_shape = update.shape
        update = update.reshape(update.size(0), -1)

    # 重要：保持 bf16，禁止 .float()
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    if update.dtype != dtype:
        update = update.to(dtype)

    if original_shape is not None:
        update = update.reshape(original_shape)

    # ---------- NorMuon ----------
    vnorm = update.norm(dim=(-2, -1), keepdim=True)

    # v_mean 与 second_momentum 同 dtype
    try:
        v_mean = (update * update).mean(dim=-1, keepdim=True, dtype=dtype)
    except TypeError:
        v_mean = (update * update).mean(dim=-1, keepdim=True)
        if v_mean.dtype != dtype:
            v_mean = v_mean.to(dtype)

    second_momentum.lerp_(v_mean, 1.0 - beta2)

    # step_size 与 eps 也用同 dtype/device
    eps_t = torch.tensor(1e-10, dtype=dtype, device=device)
    step_size = torch.rsqrt(second_momentum + eps_t)
    update.mul_(step_size)

    vnorm_new = update.norm(dim=(-2, -1), keepdim=True)
    update.mul_(vnorm / (vnorm_new + eps_t))  # 保持归一化前的范数

    # 尺度修正（标量也对齐 dtype）
    scale = max(1.0, (grad.size(-2) / grad.size(-1)) ** 0.5)
    update.mul_(torch.tensor(scale, dtype=dtype, device=device))

    return update



@torch.no_grad()
def normuon_update_rms(grad, momentum, second_momentum,
                   beta=0.95, beta2=0.95, ns_steps=5, nesterov=True):
    # 统一 dtype/device（继承 state 的 bf16）
    dtype, device = momentum.dtype, momentum.device
    g = grad if grad.dtype == dtype else grad.to(dtype)

    # 一阶动量
    momentum.lerp_(g, 1.0 - beta)

    # Nesterov：不要在 grad 上 in-place
    update = torch.lerp(g, momentum, beta) if nesterov else momentum

    # conv 权重摊平
    original_shape = None
    if update.ndim == 4:
        original_shape = update.shape
        update = update.reshape(update.size(0), -1)

    # 重要：保持 bf16，禁止 .float()
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    if update.dtype != dtype:
        update = update.to(dtype)

    if original_shape is not None:
        update = update.reshape(original_shape)

    # ---------- NorMuon ----------
    vnorm = update.norm(dim=(-2, -1), keepdim=True)

    # v_mean 与 second_momentum 同 dtype
    try:
        v_mean = (update * update).mean(dim=-1, keepdim=True, dtype=dtype)
    except TypeError:
        v_mean = (update * update).mean(dim=-1, keepdim=True)
        if v_mean.dtype != dtype:
            v_mean = v_mean.to(dtype)

    second_momentum.lerp_(v_mean, 1.0 - beta2)

    # step_size 与 eps 也用同 dtype/device
    eps_t = torch.tensor(1e-10, dtype=dtype, device=device)
    step_size = torch.rsqrt(second_momentum + eps_t)
    update.mul_(step_size)

    vnorm_new = update.norm(dim=(-2, -1), keepdim=True)
    update.mul_(vnorm / (vnorm_new + eps_t))  # 保持归一化前的范数

    # 尺度修正（标量也对齐 dtype）
    #scale = max(1.0, (grad.size(-2) / grad.size(-1)) ** 0.5)
    scale = 0.2 * (grad.size(-2) * grad.size(-1))**0.5 / (update.norm() + eps_t)
    update.mul_(torch.tensor(scale, dtype=dtype, device=device))

    return update


# modified from https://github.com/KellerJordan/Muon/blob/master/muon.py
class NorMuon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95, beta2=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, beta2=beta2)
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        params = sorted(params, key=lambda x: x.size(), reverse=True)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = group["params"]
            params_pad = params + [torch.empty_like(params[-1])] * (dist.get_world_size() - len(params) % dist.get_world_size())
            for base_i in range(len(params))[::dist.get_world_size()]:
                if base_i + dist.get_rank() < len(params):
                    p = params[base_i + dist.get_rank()]
                    had_grad = p.grad is not None
                    if not had_grad:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                        state["second_momentum_buffer"] = torch.zeros_like(p[..., 0:1])
                    update = normuon_update(p.grad, state["momentum_buffer"], state["second_momentum_buffer"], beta=group["momentum"], beta2=group["beta2"])
                    if group["weight_decay"] and had_grad:
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
                dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])

        return loss

# modified from https://github.com/KellerJordan/Muon/blob/master/muon.py
class SingleDeviceNorMuon(torch.optim.Optimizer):
    """
    Muon variant for usage in non-distributed settings.
    """
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95, beta2=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, beta2=beta2)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                had_grad = p.grad is not None
                if not had_grad:
                    # continue
                    p.grad = torch.zeros_like(p)  # Force synchronization
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                    state["second_momentum_buffer"] = torch.zeros_like(p[...,0:1])
                update = normuon_update(p.grad, state["momentum_buffer"], state["second_momentum_buffer"], beta=group["momentum"], beta2=group["beta2"])
                if group["weight_decay"] and had_grad:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])

        return loss


def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0]**step)
    buf2c = buf2 / (1 - betas[1]**step)
    return buf1c / (buf2c.sqrt() + eps)






class NorMuonWithAuxAdam(torch.optim.Optimizer):
    """
    Distributed NorMuon variant paired with an auxiliary Adam optimizer for parameters that are not
    compatible with NorMuon. Groups intended for NorMuon should set `use_muon=True`.
    """
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["beta2"] = group.get("beta2", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == {"params", "lr", "momentum", "beta2", "weight_decay", "use_muon"}
            else:
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == {"params", "lr", "betas", "eps", "weight_decay", "use_muon"}
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                params = group["params"]
                params_pad = params + [torch.empty_like(params[-1])] * (dist.get_world_size() - len(params) % dist.get_world_size())
                for base_i in range(len(params))[::dist.get_world_size()]:
                    if base_i + dist.get_rank() < len(params):
                        p = params[base_i + dist.get_rank()]
                        had_grad = p.grad is not None
                        if not had_grad:
                            p.grad = torch.zeros_like(p)
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)
                            state["second_momentum_buffer"] = torch.zeros_like(p[..., 0:1])
                        update = normuon_update(p.grad, state["momentum_buffer"], state["second_momentum_buffer"],
                                                beta=group["momentum"], beta2=group["beta2"])
                        if group["weight_decay"] and had_grad:
                            p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update.reshape(p.shape), alpha=-group["lr"])
                    dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])
            else:
                for p in group["params"]:
                    had_grad = p.grad is not None
                    if not had_grad:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    if group["weight_decay"] and had_grad:
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss



class NorMuonWithAuxAdamRMS(torch.optim.Optimizer):
    """
    Distributed NorMuon variant paired with an auxiliary Adam optimizer for parameters that are not
    compatible with NorMuon. Groups intended for NorMuon should set `use_muon=True`.
    """
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["beta2"] = group.get("beta2", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == {"params", "lr", "momentum", "beta2", "weight_decay", "use_muon"}
            else:
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == {"params", "lr", "betas", "eps", "weight_decay", "use_muon"}
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                params = group["params"]
                params_pad = params + [torch.empty_like(params[-1])] * (dist.get_world_size() - len(params) % dist.get_world_size())
                for base_i in range(len(params))[::dist.get_world_size()]:
                    if base_i + dist.get_rank() < len(params):
                        p = params[base_i + dist.get_rank()]
                        had_grad = p.grad is not None
                        if not had_grad:
                            p.grad = torch.zeros_like(p)
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)
                            state["second_momentum_buffer"] = torch.zeros_like(p[..., 0:1])
                        update = normuon_update_rms(p.grad, state["momentum_buffer"], state["second_momentum_buffer"],
                                                beta=group["momentum"], beta2=group["beta2"])
                        if group["weight_decay"] and had_grad:
                            p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update.reshape(p.shape), alpha=-group["lr"])
                    dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])
            else:
                for p in group["params"]:
                    had_grad = p.grad is not None
                    if not had_grad:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    if group["weight_decay"] and had_grad:
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss


class NorMuonGWithAuxAdamV2(torch.optim.Optimizer):
    """
    Distributed NorMuon variant paired with an auxiliary Adam optimizer for parameters that are not
    compatible with NorMuon. Groups intended for NorMuon should set `use_muon=True`.
    """
    def __init__(self, param_groups,power=0.25):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["beta2"] = group.get("beta2", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == {"params", "lr", "momentum", "beta2", "weight_decay", "use_muon"}
            else:
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == {"params", "lr", "betas", "eps", "weight_decay", "use_muon"}
        super().__init__(param_groups, dict())
        self.power=power

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                params = group["params"]
                params_pad = params + [torch.empty_like(params[-1])] * (dist.get_world_size() - len(params) % dist.get_world_size())
                for base_i in range(len(params))[::dist.get_world_size()]:
                    if base_i + dist.get_rank() < len(params):
                        p = params[base_i + dist.get_rank()]
                        had_grad = p.grad is not None
                        if not had_grad:
                            p.grad = torch.zeros_like(p)
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)
                            state["second_momentum_buffer"] = torch.zeros_like(p[..., 0:1])
                        update = normuon_generalized_ht_update_v2(p.grad, state["momentum_buffer"], state["second_momentum_buffer"],self.power,
                                                beta=group["momentum"], beta2=group["beta2"])
                        if group["weight_decay"] and had_grad:
                            p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update.reshape(p.shape), alpha=-group["lr"])
                    dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])
            else:
                for p in group["params"]:
                    had_grad = p.grad is not None
                    if not had_grad:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    if group["weight_decay"] and had_grad:
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss



class NorMuonGWithAuxAdamV2ACC(torch.optim.Optimizer):
    """
    Distributed NorMuon variant paired with an auxiliary Adam optimizer for parameters that are not
    compatible with NorMuon. Groups intended for NorMuon should set `use_muon=True`.
    """
    def __init__(self, param_groups,power=0.25):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["beta2"] = group.get("beta2", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == {"params", "lr", "momentum", "beta2", "weight_decay", "use_muon"}
            else:
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == {"params", "lr", "betas", "eps", "weight_decay", "use_muon"}
        super().__init__(param_groups, dict())
        self.power=power

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                params = group["params"]
                params_pad = params + [torch.empty_like(params[-1])] * (dist.get_world_size() - len(params) % dist.get_world_size())
                for base_i in range(len(params))[::dist.get_world_size()]:
                    if base_i + dist.get_rank() < len(params):
                        p = params[base_i + dist.get_rank()]
                        had_grad = p.grad is not None
                        if not had_grad:
                            p.grad = torch.zeros_like(p)
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)
                            state["second_momentum_buffer"] = torch.zeros_like(p[..., 0:1])
                        update = normuon_generalized_ht_update_v2_acc(p.grad, state["momentum_buffer"], state["second_momentum_buffer"],self.power,
                                                beta=group["momentum"], beta2=group["beta2"])
                        if group["weight_decay"] and had_grad:
                            p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update.reshape(p.shape), alpha=-group["lr"])
                    dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])
            else:
                for p in group["params"]:
                    had_grad = p.grad is not None
                    if not had_grad:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    if group["weight_decay"] and had_grad:
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss


class NorMuonGWithAuxAdamV2ACCV2(torch.optim.Optimizer):
    """
    Distributed NorMuon variant paired with an auxiliary Adam optimizer for parameters that are not
    compatible with NorMuon. Groups intended for NorMuon should set `use_muon=True`.
    """
    def __init__(self, param_groups,power=0.25):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["beta2"] = group.get("beta2", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == {"params", "lr", "momentum", "beta2", "weight_decay", "use_muon"}
            else:
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == {"params", "lr", "betas", "eps", "weight_decay", "use_muon"}
        super().__init__(param_groups, dict())
        self.power=power

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                params = group["params"]
                params_pad = params + [torch.empty_like(params[-1])] * (dist.get_world_size() - len(params) % dist.get_world_size())
                for base_i in range(len(params))[::dist.get_world_size()]:
                    if base_i + dist.get_rank() < len(params):
                        p = params[base_i + dist.get_rank()]
                        had_grad = p.grad is not None
                        if not had_grad:
                            p.grad = torch.zeros_like(p)
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)
                            state["second_momentum_buffer"] = torch.zeros_like(p[..., 0:1])
                        update = normuon_generalized_ht_update_v2_accv2(p.grad, state["momentum_buffer"], state["second_momentum_buffer"],self.power,
                                                beta=group["momentum"], beta2=group["beta2"])
                        if group["weight_decay"] and had_grad:
                            p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update.reshape(p.shape), alpha=-group["lr"])
                    dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])
            else:
                for p in group["params"]:
                    had_grad = p.grad is not None
                    if not had_grad:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    if group["weight_decay"] and had_grad:
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss
    

class NorMuonGWithAuxAdamV2ACCV3(torch.optim.Optimizer):
    """
    Distributed NorMuon variant paired with an auxiliary Adam optimizer for parameters that are not
    compatible with NorMuon. Groups intended for NorMuon should set `use_muon=True`.
    """
    def __init__(self, param_groups,power):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["beta2"] = group.get("beta2", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == {"params", "lr", "momentum", "beta2", "weight_decay", "use_muon"}
            else:
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == {"params", "lr", "betas", "eps", "weight_decay", "use_muon"}
        super().__init__(param_groups, dict())
        self.power=power

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                params = group["params"]
                params_pad = params + [torch.empty_like(params[-1])] * (dist.get_world_size() - len(params) % dist.get_world_size())
                for base_i in range(len(params))[::dist.get_world_size()]:
                    if base_i + dist.get_rank() < len(params):
                        p = params[base_i + dist.get_rank()]
                        had_grad = p.grad is not None
                        if not had_grad:
                            p.grad = torch.zeros_like(p)
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)
                            state["second_momentum_buffer"] = torch.zeros_like(p[..., 0:1])
                        update = normuon_generalized_ht_update_v2_accv3(p.grad, state["momentum_buffer"], state["second_momentum_buffer"],self.power,
                                                beta=group["momentum"], beta2=group["beta2"])
                        if group["weight_decay"] and had_grad:
                            p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update.reshape(p.shape), alpha=-group["lr"])
                    dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])
            else:
                for p in group["params"]:
                    had_grad = p.grad is not None
                    if not had_grad:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    if group["weight_decay"] and had_grad:
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss
    
class NorMuonGWithAuxAdam(torch.optim.Optimizer):
    """
    Distributed NorMuon variant paired with an auxiliary Adam optimizer for parameters that are not
    compatible with NorMuon. Groups intended for NorMuon should set `use_muon=True`.
    """
    def __init__(self, param_groups,power=0.25):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["beta2"] = group.get("beta2", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == {"params", "lr", "momentum", "beta2", "weight_decay", "use_muon"}
            else:
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == {"params", "lr", "betas", "eps", "weight_decay", "use_muon"}
        super().__init__(param_groups, dict())
        self.power=power

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                params = group["params"]
                params_pad = params + [torch.empty_like(params[-1])] * (dist.get_world_size() - len(params) % dist.get_world_size())
                for base_i in range(len(params))[::dist.get_world_size()]:
                    if base_i + dist.get_rank() < len(params):
                        p = params[base_i + dist.get_rank()]
                        had_grad = p.grad is not None
                        if not had_grad:
                            p.grad = torch.zeros_like(p)
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)
                            state["second_momentum_buffer"] = torch.zeros_like(p[..., 0:1])
                        update = normuon_generalized_ht_update(p.grad, state["momentum_buffer"], state["second_momentum_buffer"],self.power,
                                                beta=group["momentum"], beta2=group["beta2"])
                        if group["weight_decay"] and had_grad:
                            p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update.reshape(p.shape), alpha=-group["lr"])
                    dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])
            else:
                for p in group["params"]:
                    had_grad = p.grad is not None
                    if not had_grad:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    if group["weight_decay"] and had_grad:
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss


class NorMuonGWithAuxAdamRMS(torch.optim.Optimizer):
    """
    Distributed NorMuon variant paired with an auxiliary Adam optimizer for parameters that are not
    compatible with NorMuon. Groups intended for NorMuon should set `use_muon=True`.
    """
    def __init__(self, param_groups,power=0.25):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["beta2"] = group.get("beta2", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == {"params", "lr", "momentum", "beta2", "weight_decay", "use_muon"}
            else:
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == {"params", "lr", "betas", "eps", "weight_decay", "use_muon"}
        super().__init__(param_groups, dict())
        self.power=power

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                params = group["params"]
                params_pad = params + [torch.empty_like(params[-1])] * (dist.get_world_size() - len(params) % dist.get_world_size())
                for base_i in range(len(params))[::dist.get_world_size()]:
                    if base_i + dist.get_rank() < len(params):
                        p = params[base_i + dist.get_rank()]
                        had_grad = p.grad is not None
                        if not had_grad:
                            p.grad = torch.zeros_like(p)
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)
                            state["second_momentum_buffer"] = torch.zeros_like(p[..., 0:1])
                        update = normuon_generalized_ht_update_rms(p.grad, state["momentum_buffer"], state["second_momentum_buffer"],self.power,
                                                beta=group["momentum"], beta2=group["beta2"])
                        if group["weight_decay"] and had_grad:
                            p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update.reshape(p.shape), alpha=-group["lr"])
                    dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])
            else:
                for p in group["params"]:
                    had_grad = p.grad is not None
                    if not had_grad:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    if group["weight_decay"] and had_grad:
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss


class SingleDeviceNorMuonWithAuxAdam(torch.optim.Optimizer):
    """
    Non-distributed counterpart to NorMuonWithAuxAdam.
    """
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["beta2"] = group.get("beta2", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == {"params", "lr", "momentum", "beta2", "weight_decay", "use_muon"}
            else:
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == {"params", "lr", "betas", "eps", "weight_decay", "use_muon"}
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    had_grad = p.grad is not None
                    if not had_grad:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                        state["second_momentum_buffer"] = torch.zeros_like(p[..., 0:1])
                    update = normuon_update(p.grad, state["momentum_buffer"], state["second_momentum_buffer"],
                                            beta=group["momentum"], beta2=group["beta2"])
                    if group["weight_decay"] and had_grad:
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    had_grad = p.grad is not None
                    if not had_grad:
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    if group["weight_decay"] and had_grad:
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss
