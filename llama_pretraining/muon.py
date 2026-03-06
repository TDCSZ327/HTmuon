import torch
import torch.distributed as dist
import math

def zeropower_via_newtonschulz5(G, steps: int):
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


    X = G.float()

 
    transposed = False
    if X.size(-2) < X.size(-1):
        X = X.mT
        transposed = True


    frob = X.norm(dim=(-2, -1), keepdim=True)
    X = X / (frob + eps)

    n = X.size(-1)
    I = torch.eye(n, device=X.device, dtype=X.dtype)


    for _ in range(steps):
        XtX = X.mT @ X                    # (..., n, n)
        X = 0.5 * X @ (3.0 * I - XtX)     # (..., m, n)

    if transposed:
        X = X.mT


    return X.to(G.dtype)


def gram_root_1_16_via_newtonschulz(G, power, ns_steps: int = 5, eps: float = 1e-6):
    """
    Approximate (G^T G)^{1/16} using 4x Newton–Schulz matrix square root iterations.

    G: (..., m, n)
    return: (..., n, n) SPD matrix approximating (G^T G)^{1/16}
    """
    assert G.ndim >= 2

 
    G_work = G.float()


    A = G_work.transpose(-2, -1) @ G_work


    n = A.size(-1)
    I_base = torch.eye(n, device=A.device, dtype=A.dtype)
    I = I_base.expand(A.shape)  
    A = A + eps * I

    S = A   
    val = math.log(power, 0.5) + 1   
    val = int(round(val)) 

    for _ in range(val):  
        
        alpha = S.norm(dim=(-2, -1), keepdim=True) 
        alpha = alpha.clamp_min(eps)               
        S_tilde = S / alpha

        # 2) Newton–Schulz for matrix square root of S_tilde
        Y = S_tilde.clone()
        Z = I_base.expand(S_tilde.shape).clone()    

        for _ in range(ns_steps):
            T = 3.0 * I - Z @ Y          # 3I - Z Y
            Y = 0.5 * (Y @ T)            # Y_{k+1}
            Z = 0.5 * (T @ Z)            # Z_{k+1}

       
        S = (alpha.sqrt()) * Y          

        
        S = 0.5 * (S + S.transpose(-2, -1))         # symmetrize
        S = S + eps * I


    return S.to(G.dtype)


import torch
import math

def ns_matrix_sqrt(S, ns_steps: int = 5, eps: float = 1e-6):

    
    assert S.ndim >= 2
    n = S.size(-1)
    I_base = torch.eye(n, device=S.device, dtype=S.dtype)
    I = I_base.expand(S.shape)


    alpha = S.norm(dim=(-2, -1), keepdim=True).clamp_min(eps)
    S_tilde = S / alpha

    Y = S_tilde.clone()
    Z = I_base.expand(S_tilde.shape).clone()

    for _ in range(ns_steps):
        T = 3.0 * I - Z @ Y
        Y = 0.5 * (Y @ T)
        Z = 0.5 * (T @ Z)

    S_sqrt = alpha.sqrt() * Y
    S_sqrt = 0.5 * (S_sqrt + S_sqrt.transpose(-2, -1)) + eps * I
    return S_sqrt


def ns_matrix_invsqrt(S, ns_steps: int = 5, eps: float = 1e-6):
    
    assert S.ndim >= 2
    n = S.size(-1)
    I_base = torch.eye(n, device=S.device, dtype=S.dtype)
    I = I_base.expand(S.shape)

    alpha = S.norm(dim=(-2, -1), keepdim=True).clamp_min(eps)
    S_tilde = S / alpha

    Y = S_tilde.clone()
    Z = I_base.expand(S_tilde.shape).clone()

    for _ in range(ns_steps):
        T = 3.0 * I - Z @ Y
        Y = 0.5 * (Y @ T)
        Z = 0.5 * (T @ Z)

    
    S_invsqrt = alpha.rsqrt() * Z
    S_invsqrt = 0.5 * (S_invsqrt + S_invsqrt.transpose(-2, -1))
    
    return S_invsqrt

def gram_power_minus_7_16_via_newtonschulz(
    G,
    ns_steps: int = 5,
    eps: float = 1e-6,
):
    
    assert G.ndim >= 2

    
    G_work = G.float()

    
    A = G_work.transpose(-2, -1) @ G_work
    n = A.size(-1)
    I_base = torch.eye(n, device=A.device, dtype=A.dtype)
    I = I_base.expand(A.shape)

   
    A = A + eps * I

    
    S = A
    for _ in range(3):
        S = ns_matrix_sqrt(S, ns_steps=ns_steps, eps=eps)

    
    A_neg_1_16 = ns_matrix_invsqrt(S, ns_steps=ns_steps, eps=eps)


    A_neg_7_16 = torch.linalg.matrix_power(A_neg_1_16, 7)

   
    A_neg_7_16 = 0.5 * (A_neg_7_16 + A_neg_7_16.transpose(-2, -1)) + eps * I

    return A_neg_7_16.to(G.dtype)



def taylor_sqrt_poly(S, I_base, degree: int = 4, eps: float = 1e-6):

    I = I_base.expand(S.shape)

    
    alpha = S.norm(dim=(-2, -1), keepdim=True)   
    alpha = alpha.clamp_min(eps)
    S_tilde = S / alpha                           


    X = S_tilde - I


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

    
    S_sqrt = (alpha.sqrt()) * Y

    return S_sqrt


def gram_root_1_16_via_taylor(G, power: float = 0.125, taylor_degree: int = 4,
                              eps: float = 1e-6):

    assert G.ndim >= 2

    
    G_work = G.float()

    
    A = G_work.transpose(-2, -1) @ G_work

    
    n = A.size(-1)
    I_base = torch.eye(n, device=A.device, dtype=A.dtype)
    I = I_base.expand(A.shape)
    A = A + eps * I

    
    S = A

    
    steps = math.log(power, 0.5) + 1  
    steps = int(round(steps))                     

    for i in range(steps):
        
        S = taylor_sqrt_poly(S, I_base, degree=taylor_degree, eps=eps)

        
        S = 0.5 * (S + S.transpose(-2, -1))
        S = S + eps * I
        

    
    return S.to(G.dtype)






def msign_ht(g: torch.Tensor) -> torch.Tensor:
    orig_dtype = g.dtype
    U32, S32, Vh32 = torch.linalg.svd(g.to(torch.float32), full_matrices=False)
    k = S32.shape[-1]  
    
    d = 1.0 / torch.sqrt(torch.arange(1, k + 1, device=S32.device, dtype=S32.dtype))
    
    d_broadcast = d.view(*([1] * (U32.ndim - 2)), 1, k)
    out32 = (U32 * d_broadcast) @ Vh32
    return out32.to(orig_dtype)



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

    
    Sp32 = S32.pow(p)

    
    Sp32_view = Sp32.view(*([1] * (U32.ndim - 2)), 1, k)
    out32 = (U32 * Sp32_view) @ Vh32

    return out32.to(orig_dtype)

def svd_project_uvt(g: torch.Tensor, p: float = 0.25) -> torch.Tensor:
    orig_dtype = g.dtype
    U32, S32, Vh32 = torch.linalg.svd(g.to(torch.float32), full_matrices=False)

    out32 = U32 @ Vh32

    return out32.to(orig_dtype)

def msign3_ht(g: torch.Tensor) -> torch.Tensor:
    orig_dtype = g.dtype
    U32, S32, Vh32 = torch.linalg.svd(g.to(torch.float32), full_matrices=False)
    k = S32.shape[-1]  
    # d = [1/sqrt(1), 1/sqrt(2), ..., 1/sqrt(k)]
    d = 1.0 / torch.arange(1, k + 1, device=S32.device, dtype=S32.dtype)
    
    d_broadcast = d.view(*([1] * (U32.ndim - 2)), 1, k)
    out32 = (U32 * d_broadcast) @ Vh32
    return out32.to(orig_dtype)

def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4: # for the case of conv filters
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update


def rnnp_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4: # for the case of conv filters
        update = update.view(len(update), -1)
    eps = 1e-12
    if update.ndim == 1:
        denom = update.norm() + eps
    else:
        denom = update.norm(dim=-1, keepdim=True) + eps
    update = update / denom
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update







def muon2_ht_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4: # for the case of conv filters
        update = update.view(len(update), -1)
    update = msign_ht(update)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update


def muon3_ht_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4: # for the case of conv filters
        update = update.view(len(update), -1)
    update = msign3_ht(update)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update




def muon_generalized_ht_update(grad, momentum, power=0.25, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4: # for the case of conv filters
        update = update.view(len(update), -1)
    update = msign_generalized_ht(update,power)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update


def svd_project_uvt_update(grad, momentum, power=0.25, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4: # for the case of conv filters
        update = update.view(len(update), -1)
    update = svd_project_uvt(update,power)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update


def muon_generalized_ht_update_v2(grad, momentum, power=0.25, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4: # for the case of conv filters
        update = update.view(len(update), -1)
    update = msign_generalized_ht_v2(update,power)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update


def muon_generalized_ht_update_v2_acc(grad, momentum, power=0.125, beta=0.95, ns_steps=5,ns_steps_2=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4: # for the case of conv filters
        update = update.view(len(update), -1)
    update_1 = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update_2 = gram_root_1_16_via_newtonschulz(update, power,ns_steps=ns_steps_2)
    update = update_1 @ update_2
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update




def muon_generalized_ht_update_v2_accv2(grad, momentum, power=0.125, beta=0.95, ns_steps=10, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4: # for the case of conv filters
        update = update.view(len(update), -1)
    update_1 = zeropower_via_newtonschulz_exact(update, steps=ns_steps)
    update_2 = gram_root_1_16_via_newtonschulz(update, power,ns_steps=ns_steps)
    update = update_1 @ update_2
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update

def muon_generalized_ht_update_v2_accv3(grad, momentum, power, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4: # for the case of conv filters
        update = update.view(len(update), -1)
    update_1 = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update_2 = gram_root_1_16_via_taylor(update, power)
    update = update_1 @ update_2
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update



def muon_generalized_ht_update_v2_accv4(grad, momentum, power=0.125, beta=0.95, ns_steps=5,ns_steps_2=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4: # for the case of conv filters
        update = update.view(len(update), -1)
    update_2 = gram_power_minus_7_16_via_newtonschulz(update)
    update = update @ update_2
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update



_DEFAULT_DTYPE = torch.bfloat16  


def sym(x: torch.Tensor) -> torch.Tensor:
    """Symmetrize (use transpose, not conjugate)"""
    return (x + x.T) * x.new_tensor(0.5)

def skew(x: torch.Tensor) -> torch.Tensor:
    """Skew-symmetrize"""
    return (x - x.T) * x.new_tensor(0.5)

def proj(g: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Projection onto the orthogonal tangent space: g - w @ sym(w^T g)"""
    return g - w @ sym(w.T @ g)

def _maybe_real(x: torch.Tensor, tol: float = 1e-7) -> torch.Tensor:
    """
    If result is complex but the imaginary part is tiny (numerical noise),
    take the real part; then cast back to the original real dtype (e.g., bf16).
    """
    if torch.is_complex(x):
        im_max = x.imag.abs().max()
        re_max = x.real.abs().max().clamp_min(1.0)
        if float(im_max / re_max) < tol:
            x = x.real
    return x


def mcsgn(x: torch.Tensor) -> torch.Tensor:
    """
    Matrix sign via eigen-decomposition:
      mcsgn(X) = V diag(sign(λ)) V^{-1}

    - Input/Output: same dtype as x (bf16)
    - Internally: upcast to float32 for eig, then cast back
    - Note:
      * torch.linalg.eig returns complex eigenvalues/vectors even for real inputs.
      * torch.sign on complex numbers returns z/|z| (not "sign of real part").
        This is more general but differs from some definitions based on Re(λ).
    """
    orig_dtype = x.dtype
    x32 = x.to(torch.float32)
    eigvals, eigvecs = torch.linalg.eig(x32)   
    sgn = torch.sign(eigvals)                 
    D = torch.diag(sgn)
    invV = torch.linalg.inv(eigvecs)
    out = eigvecs @ D @ invV                  
    out = _maybe_real(out).to(torch.float32)
    return out.to(orig_dtype)

def msign(g: torch.Tensor) -> torch.Tensor:
    """
    Matrix sign via SVD:
      msign(G) = U diag(sign(σ)) Vh

    - Input/Output: same dtype as g (bf16)
    - Internally: upcast to float32 for SVD, then cast back
    """
    orig_dtype = g.dtype
    U32, S32, Vh32 = torch.linalg.svd(g.to(torch.float32), full_matrices=False)
    D32 = torch.diag(torch.sign(S32))
    out32 = U32 @ D32 @ Vh32
    return out32.to(orig_dtype)


@torch.no_grad()
def stiefel_by_svd(
    g: torch.Tensor,
    w: torch.Tensor,
    steps: int = 20,
    dtype: torch.dtype = _DEFAULT_DTYPE,
    verbose: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Iteration constructed via SVD (bf16 externally; fp32 internally where needed).
    """
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    g = g.to(dev, dtype=dtype)
    w = w.to(dev, dtype=dtype)

    x = -sym((w.T @ g))

    for i in range(1, steps + 1):
        z = g + w @ x                               # bf16
        U32, S32, Vh32 = torch.linalg.svd(z.to(torch.float32), full_matrices=False)
        phi32 = (U32 * torch.sign(S32)) @ Vh32      # fp32
        phi = phi32.to(z.dtype)                     # back to bf16

        if verbose:
            ip = (phi.to(torch.float32) * g.to(torch.float32)).sum().item()
            te = torch.abs(sym((w.T @ phi).to(torch.float32))).mean().item()
            print(f'step: {i}, inner product: {ip}, tangent error: {te}')

        if i == steps:
            return phi

        # === Update x in fp32, then cast back to bf16 ===
        # For complex-safe handling: V = Vh.mH; for real inputs it equals Vh.T
        V32 = Vh32.mH                                # (m, m)
        M32 = sym((z.T.to(torch.float32) @ phi32 @ (w.T.to(torch.float32) @ g.to(torch.float32))))
        A32 = Vh32 @ M32 @ V32

        Ssum32 = S32.unsqueeze(0) + S32.unsqueeze(1)
        A32 = A32 / (Ssum32 + eps)                  # avoid divide-by-zero

        x32 = -2.0 * V32 @ A32 @ Vh32
        x = x32.to(dtype)                           # back to bf16




@torch.no_grad()
def stiefel_by_svd_2(
    g: torch.Tensor,
    w: torch.Tensor,
    steps: int = 20,
    dtype: torch.dtype = _DEFAULT_DTYPE,
    verbose: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Iteration constructed via SVD (bf16 externally; fp32 internally where needed).
    """
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    g = g.to(dev, dtype=dtype)
    w = w.to(dev, dtype=dtype)

    x = -sym((g@w.T))


    for i in range(1, steps + 1):
        z = g + x @ w                               # bf16
        U32, S32, Vh32 = torch.linalg.svd(z.to(torch.float32), full_matrices=False)
        phi32 = (U32 * torch.sign(S32)) @ Vh32      # fp32
        phi = phi32.to(z.dtype)                     # back to bf16

        if verbose:
            ip = (phi.to(torch.float32) * g.to(torch.float32)).sum().item()
            te = torch.abs(sym((w.T @ phi).to(torch.float32))).mean().item()
            print(f'step: {i}, inner product: {ip}, tangent error: {te}')

        if i == steps:
            return phi

        # === Update x in fp32, then cast back to bf16 ===
        # For complex-safe handling: V = Vh.mH; for real inputs it equals Vh.T
        V32 = U32.mH                                # (m, m)
        M32 = sym((phi32 @ z.T.to(torch.float32) @ (g.to(torch.float32) @ w.T.to(torch.float32))))
        A32 = U32 @ M32 @ V32

        Ssum32 = S32.unsqueeze(0) + S32.unsqueeze(1)
        A32 = A32 / (Ssum32 + eps)                  # avoid divide-by-zero

        x32 = -2.0 * V32 @ A32 @ U32
        x = x32.to(dtype)                           # back to bf16


@torch.no_grad()
def HT_by_svd(
    g: torch.Tensor,
    w: torch.Tensor,
    Sigma_neg_half: torch.Tensor,
    steps: int = 20,
    dtype: torch.dtype = _DEFAULT_DTYPE,
    verbose: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Iteration constructed via SVD (bf16 externally; fp32 internally where needed).
    """
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    g = g.to(dev, dtype=dtype)
    w = w.to(dev, dtype=dtype)
    w = w@ Sigma_neg_half

    x = -sym((w.T @ g))

    for i in range(1, steps + 1):
        z = g + w @ x                               # bf16
        U32, S32, Vh32 = torch.linalg.svd(z.to(torch.float32), full_matrices=False)
        phi32 = (U32 * torch.sign(S32)) @ Vh32      # fp32
        phi = phi32.to(z.dtype)                     # back to bf16

        if verbose:
            ip = (phi.to(torch.float32) * g.to(torch.float32)).sum().item()
            te = torch.abs(sym((w.T @ phi).to(torch.float32))).mean().item()
            print(f'step: {i}, inner product: {ip}, tangent error: {te}')

        if i == steps:
            return phi

        # === Update x in fp32, then cast back to bf16 ===
        # For complex-safe handling: V = Vh.mH; for real inputs it equals Vh.T
        V32 = Vh32.mH                                # (m, m)
        M32 = sym((z.T.to(torch.float32) @ phi32 @ (w.T.to(torch.float32) @ g.to(torch.float32))))
        A32 = Vh32 @ M32 @ V32

        Ssum32 = S32.unsqueeze(0) + S32.unsqueeze(1)
        A32 = A32 / (Ssum32 + eps)                  # avoid divide-by-zero

        x32 = -2.0 * V32 @ A32 @ Vh32
        x = x32.to(dtype) 


@torch.no_grad()
def HT_by_svd_2(
    g: torch.Tensor,
    w: torch.Tensor,
    Sigma_neg_half: torch.Tensor,
    steps: int = 20,
    dtype: torch.dtype = _DEFAULT_DTYPE,
    verbose: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Iteration constructed via SVD (bf16 externally; fp32 internally where needed).
    """
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    g = g.to(dev, dtype=dtype)
    w = w.to(dev, dtype=dtype)
    w=Sigma_neg_half@w

    x = -sym((g@w.T))

    for i in range(1, steps + 1):
        z = g + x @ w                               # bf16
        U32, S32, Vh32 = torch.linalg.svd(z.to(torch.float32), full_matrices=False)
        phi32 = (U32 * torch.sign(S32)) @ Vh32      # fp32
        phi = phi32.to(z.dtype)                     # back to bf16

        if verbose:
            ip = (phi.to(torch.float32) * g.to(torch.float32)).sum().item()
            te = torch.abs(sym((w.T @ phi).to(torch.float32))).mean().item()
            print(f'step: {i}, inner product: {ip}, tangent error: {te}')

        if i == steps:
            return phi

        # === Update x in fp32, then cast back to bf16 ===
        # For complex-safe handling: V = Vh.mH; for real inputs it equals Vh.T
        V32 = U32.mH                                # (m, m)
        M32 = sym((phi32 @ z.T.to(torch.float32) @ (g.to(torch.float32) @ w.T.to(torch.float32))))
        A32 = U32 @ M32 @ V32

        Ssum32 = S32.unsqueeze(0) + S32.unsqueeze(1)
        A32 = A32 / (Ssum32 + eps)                  # avoid divide-by-zero

        x32 = -2.0 * V32 @ A32 @ U32
        x = x32.to(dtype)                           # back to bf16



def muon_orth_update(param,grad,lr, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4: # for the case of conv filters
        update = update.view(len(update), -1)
    if update.shape[0]==update.shape[1]:
        skew=(param.T@update-update.T@param)*(update.T@param).new_tensor(0.5)
        O=zeropower_via_newtonschulz5(skew, steps=ns_steps)
        I = torch.eye(O.size(0), device=O.device, dtype=O.dtype)
        update=(I-lr*O)@(I-O.t()@O+O.t()@O*(O.t()@O).new_tensor(1.0 / (1.0 + lr*lr)).sqrt())
    
    elif update.shape[0]>update.shape[1]:
        update= stiefel_by_svd(update,param,steps=ns_steps*2)
        update=update*update.new_tensor(1.0 / (1.0 + lr*lr)).sqrt()
    else:
        update= stiefel_by_svd_2(update,param,steps=ns_steps*2)
        update=update*update.new_tensor(1.0 / (1.0 + lr*lr)).sqrt()
    return update



def muon_ht_update(param,grad,lr, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4: # for the case of conv filters
        update = update.view(len(update), -1)
    if update.shape[0]==update.shape[1]:

        i = torch.arange(1,  update.shape[0]+ 1, dtype=param.dtype, device=param.device)
        diag = i.reciprocal()  

        Sigma = torch.diag(diag)                     # A = diag(1/i)
        Sigma_half = torch.diag(torch.sqrt(diag))    # A^{1/2} = diag((1/i)^{1/2})
        Sigma_neg_half = torch.diag(torch.rsqrt(diag)) # A^{-1/2} = diag((1/i)^{-1/2}) = diag(sqrt(i))

        Sigma_inv = torch.diag(i) 


        skew=((param@Sigma_neg_half).T@update-update.T@(param@Sigma_neg_half))*(update.T@(param@Sigma_neg_half)).new_tensor(0.5)
        O=zeropower_via_newtonschulz5(skew, steps=ns_steps)
        I = torch.eye(O.size(0), device=O.device, dtype=O.dtype)
        update=(I-lr*O@Sigma_neg_half)@(I-Sigma_inv@O.t()@O+Sigma_inv@O.t()@O*(Sigma_inv@O.t()@O).new_tensor(1.0 / (1.0 + lr*lr)).sqrt())
    
    elif update.shape[0]>update.shape[1]:
        i = torch.arange(1,  update.shape[1]+ 1, dtype=param.dtype, device=param.device)
        diag = i.reciprocal()  

        Sigma = torch.diag(diag)                     # A = diag(1/i)
        Sigma_half = torch.diag(torch.sqrt(diag))    # A^{1/2} = diag((1/i)^{1/2})
        Sigma_neg_half = torch.diag(torch.rsqrt(diag)) # A^{-1/2} = diag((1/i)^{-1/2}) = diag(sqrt(i))

        Sigma_inv = torch.diag(i) 
        update= HT_by_svd(update,param, Sigma_neg_half,steps=ns_steps*2)
        update=update*update.new_tensor(1.0 / (1.0 + lr*lr)).sqrt()
    else:
        i = torch.arange(1,  update.shape[0]+ 1, dtype=param.dtype, device=param.device)
        diag = i.reciprocal()  

        Sigma = torch.diag(diag)                     # A = diag(1/i)
        Sigma_half = torch.diag(torch.sqrt(diag))    # A^{1/2} = diag((1/i)^{1/2})
        Sigma_neg_half = torch.diag(torch.rsqrt(diag)) # A^{-1/2} = diag((1/i)^{-1/2}) = diag(sqrt(i))

        Sigma_inv = torch.diag(i) 
        update= HT_by_svd_2(update,param, Sigma_neg_half,steps=ns_steps*2)
        update=update*update.new_tensor(1.0 / (1.0 + lr*lr)).sqrt()
    return update


    







class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. For efficient orthogonalization we use a Newton-Schulz iteration, which has the
    advantage that it can be stably run in bfloat16 on the GPU.

    Muon should only be used for hidden weight layers. The input embedding, final output layer,
    and any internal gains or biases should be optimized using a standard method such as AdamW.
    Hidden convolutional weights can be trained using Muon by viewing them as 2D and then
    collapsing their last 3 dimensions.

    Arguments:
        lr: The learning rate, in units of spectral norm per update.
        weight_decay: The AdamW-style weight decay.
        momentum: The momentum. A value of 0.95 here is usually fine.
    """
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
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
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
                dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])

        return loss


class SingleDeviceMuon(torch.optim.Optimizer):
    """
    Muon variant for usage in non-distributed settings.
    """
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    # continue
                    p.grad = torch.zeros_like(p)  # Force synchronization
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])

        return loss


def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0]**step)
    buf2c = buf2 / (1 - betas[1]**step)
    return buf1c / (buf2c.sqrt() + eps)



class RNNP(torch.optim.Optimizer):
    """
    Distributed Muon variant that can be used for all parameters in the network, since it runs an
    internal AdamW for the parameters that are not compatible with Muon. The user must manually
    specify which parameters shall be optimized with Muon and which with Adam by passing in a
    list of param_groups with the `use_muon` flag set.

    The point of this class is to allow the user to have a single optimizer in their code, rather
    than having both a Muon and an Adam which each need to be stepped.

    You can see an example usage below:

    https://github.com/KellerJordan/modded-nanogpt/blob/master/records/052525_MuonWithAuxAdamExample/b01550f9-03d8-4a9c-86fe-4ab434f1c5e0.txt#L470
    ```
    hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n]
    embed_params = [p for n, p in model.named_parameters() if "embed" in n]
    scalar_params = [p for p in model.parameters() if p.ndim < 2]
    head_params = [model.lm_head.weight]

    from muon import MuonWithAuxAdam
    adam_groups = [dict(params=head_params, lr=0.22), dict(params=embed_params, lr=0.6), dict(params=scalar_params, lr=0.04)]
    adam_groups = [dict(**g, betas=(0.8, 0.95), eps=1e-10, use_muon=False) for g in adam_groups]
    muon_group = dict(params=hidden_matrix_params, lr=0.05, momentum=0.95, use_muon=True)
    param_groups = [*adam_groups, muon_group]
    optimizer = MuonWithAuxAdam(param_groups)
    ```
    """
    def __init__(self, param_groups):

        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon"])
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
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
                        if p.grad is None:
                            # continue
                            p.grad = torch.zeros_like(p)  # Force synchronization
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)
                        update = rnnp_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update.reshape(p.shape), alpha=-group["lr"])
                    dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss




class MuonWithAuxAdam(torch.optim.Optimizer):
    """
    Distributed Muon variant that can be used for all parameters in the network, since it runs an
    internal AdamW for the parameters that are not compatible with Muon. The user must manually
    specify which parameters shall be optimized with Muon and which with Adam by passing in a
    list of param_groups with the `use_muon` flag set.

    The point of this class is to allow the user to have a single optimizer in their code, rather
    than having both a Muon and an Adam which each need to be stepped.

    You can see an example usage below:

    https://github.com/KellerJordan/modded-nanogpt/blob/master/records/052525_MuonWithAuxAdamExample/b01550f9-03d8-4a9c-86fe-4ab434f1c5e0.txt#L470
    ```
    hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n]
    embed_params = [p for n, p in model.named_parameters() if "embed" in n]
    scalar_params = [p for p in model.parameters() if p.ndim < 2]
    head_params = [model.lm_head.weight]

    from muon import MuonWithAuxAdam
    adam_groups = [dict(params=head_params, lr=0.22), dict(params=embed_params, lr=0.6), dict(params=scalar_params, lr=0.04)]
    adam_groups = [dict(**g, betas=(0.8, 0.95), eps=1e-10, use_muon=False) for g in adam_groups]
    muon_group = dict(params=hidden_matrix_params, lr=0.05, momentum=0.95, use_muon=True)
    param_groups = [*adam_groups, muon_group]
    optimizer = MuonWithAuxAdam(param_groups)
    ```
    """
    def __init__(self, param_groups):

        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon"])
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
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
                        if p.grad is None:
                            # continue
                            p.grad = torch.zeros_like(p)  # Force synchronization
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)
                        update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update.reshape(p.shape), alpha=-group["lr"])
                    dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss



class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. For efficient orthogonalization we use a Newton-Schulz iteration, which has the
    advantage that it can be stably run in bfloat16 on the GPU.

    Muon should only be used for hidden weight layers. The input embedding, final output layer,
    and any internal gains or biases should be optimized using a standard method such as AdamW.
    Hidden convolutional weights can be trained using Muon by viewing them as 2D and then
    collapsing their last 3 dimensions.

    Arguments:
        lr: The learning rate, in units of spectral norm per update.
        weight_decay: The AdamW-style weight decay.
        momentum: The momentum. A value of 0.95 here is usually fine.
    """
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
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
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
                dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])

        return loss


class SingleDeviceMuon(torch.optim.Optimizer):
    """
    Muon variant for usage in non-distributed settings.
    """
    def __init__(self, params, lr=0.02, weight_decay=0, momentum=0.95):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    # continue
                    p.grad = torch.zeros_like(p)  # Force synchronization
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha=-group["lr"])

        return loss


def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0]**step)
    buf2c = buf2 / (1 - betas[1]**step)
    return buf1c / (buf2c.sqrt() + eps)







class HTMuonHTWithAuxAdam(torch.optim.Optimizer):

    def __init__(self, param_groups,power=0.25):

        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon"])
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
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
                        if p.grad is None:
                            # continue
                            p.grad = torch.zeros_like(p)  # Force synchronization
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)
                        update = muon_generalized_ht_update(p.grad, state["momentum_buffer"],self.power, beta=group["momentum"])
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update.reshape(p.shape), alpha=-group["lr"])
                    dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss







class HTMuonWithAuxAdam(torch.optim.Optimizer):

    def __init__(self, param_groups,power=0.25):

        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon"])
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
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
                        if p.grad is None:
                            # continue
                            p.grad = torch.zeros_like(p)  # Force synchronization
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)
                        update = muon_generalized_ht_update_v2(p.grad, state["momentum_buffer"],self.power, beta=group["momentum"])
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update.reshape(p.shape), alpha=-group["lr"])
                    dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss



class HTMuonIntervalWithAuxAdam(torch.optim.Optimizer):


    def __init__(self, param_groups, power=0.25, interval=10, start_step=0):
        if interval <= 0:
            raise ValueError(f"`interval` must be positive, got {interval}")

        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon"])
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])

        super().__init__(param_groups, dict())

        self.power = power
        self.interval = interval
        
        self.step_idx = start_step

    @property
    def global_step(self):
        
        return self.step_idx

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

       
        self.step_idx += 1
        use_generalized = (self.step_idx % self.interval == 0)

        for group in self.param_groups:
            if group["use_muon"]:
                params = group["params"]
               
                world_size = dist.get_world_size()
                pad_len = (world_size - len(params) % world_size) % world_size
                params_pad = params + [torch.empty_like(params[-1])] * pad_len

                for base_i in range(len(params))[::world_size]:
                    if base_i + dist.get_rank() < len(params):
                        p = params[base_i + dist.get_rank()]
                        if p.grad is None:
                            # Force synchronization
                            p.grad = torch.zeros_like(p)
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)

                        if use_generalized:
                            
                            update = muon_generalized_ht_update_v2(
                                p.grad,
                                state["momentum_buffer"],
                                self.power,
                                beta=group["momentum"],
                            )
                        else:
                            
                            update = muon_update(
                                p.grad,
                                state["momentum_buffer"],
                                beta=group["momentum"],
                            )

                        p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update.reshape(p.shape), alpha=-group["lr"])

                    dist.all_gather(
                        params_pad[base_i:base_i + world_size],
                        params_pad[base_i + dist.get_rank()],
                    )
            else:
                
                for p in group["params"]:
                    if p.grad is None:
                        # Force synchronization
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(
                        p.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        group["betas"],
                        group["eps"],
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss




class HTMuonNSWithAuxAdam(torch.optim.Optimizer):

    def __init__(self, param_groups,power=0.25):

        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon"])
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
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
                        if p.grad is None:
                            # continue
                            p.grad = torch.zeros_like(p)  # Force synchronization
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)
                        update = muon_generalized_ht_update_v2_acc(p.grad, state["momentum_buffer"],self.power, beta=group["momentum"])
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update.reshape(p.shape), alpha=-group["lr"])
                    dist.all_gather(params_pad[base_i:base_i + dist.get_world_size()], params_pad[base_i + dist.get_rank()])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss



class HTMuonNSIntervalWithAuxAdam(torch.optim.Optimizer):


    def __init__(self, param_groups, power=0.25, interval=10, start_step=0):
        if interval <= 0:
            raise ValueError(f"`interval` must be positive, got {interval}")

        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                group["params"] = sorted(group["params"], key=lambda x: x.size(), reverse=True)
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon"])
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])

        super().__init__(param_groups, dict())

        self.power = power
        self.interval = interval
       
        self.step_idx = start_step

    @property
    def global_step(self):
        
        return self.step_idx

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        
        self.step_idx += 1
        use_generalized = (self.step_idx % self.interval == 0)

        for group in self.param_groups:
            if group["use_muon"]:
                params = group["params"]
                
                world_size = dist.get_world_size()
                pad_len = (world_size - len(params) % world_size) % world_size
                params_pad = params + [torch.empty_like(params[-1])] * pad_len

                for base_i in range(len(params))[::world_size]:
                    if base_i + dist.get_rank() < len(params):
                        p = params[base_i + dist.get_rank()]
                        if p.grad is None:
                            # Force synchronization
                            p.grad = torch.zeros_like(p)
                        state = self.state[p]
                        if len(state) == 0:
                            state["momentum_buffer"] = torch.zeros_like(p)

                        if use_generalized:
                            
                            update =  muon_generalized_ht_update_v2_acc(p.grad, state["momentum_buffer"],self.power, beta=group["momentum"])
                        else:
                            
                            update = muon_update(
                                p.grad,
                                state["momentum_buffer"],
                                beta=group["momentum"],
                            )

                        p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update.reshape(p.shape), alpha=-group["lr"])

                    dist.all_gather(
                        params_pad[base_i:base_i + world_size],
                        params_pad[base_i + dist.get_rank()],
                    )
            else:
                
                for p in group["params"]:
                    if p.grad is None:
                        # Force synchronization
                        p.grad = torch.zeros_like(p)
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(
                        p.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        group["betas"],
                        group["eps"],
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss






        
class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    """
    Non-distributed variant of MuonWithAuxAdam.
    """
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon"])
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
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
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss
