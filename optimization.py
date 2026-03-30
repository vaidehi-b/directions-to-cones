import torch
import random

from torch import Tensor
from typing import Callable
from jaxtyping import Float
import torch.nn.functional as F

from utils import run_with_ablation, run_with_steering


# ---- Loss functions ----

def compute_ce_loss(logits, targets):
    return F.cross_entropy(logits, targets)


def compute_kl(p_logits, q_logits, reduction="batchmean"):
    p = F.log_softmax(p_logits, dim=-1)
    q = F.softmax(q_logits, dim=-1)
    return F.kl_div(p, q, reduction=reduction)


# ---- RDO: Refusal Direction Optimization ----

def refusal_direction_optimization(model, tokenizer, data, coef, alpha=1.0, eta=0.01, lambda_abl=1.0, lambda_add=1.0, lambda_ret=1.0, num_steps=10):

    # initialize refusal direction
    r = torch.randn(model.cfg.d_model, device="cuda", dtype=torch.float32)
    r = r / r.norm()
    r.requires_grad_()
    optimizer = torch.optim.Adam([r], lr=eta)

    for step in range(num_steps):
        batch = random.choice(data)
        p_harm, p_safe, t_answer, t_refusal, t_retain = batch

        # ce loss between post-ablation output and target response (harmful)
        ablated_output = run_with_ablation(model, tokenizer, r, p_harm, type='logits')
        loss_abl = compute_ce_loss(ablated_output, t_answer)

        # ce loss between post-steering output and target response
        added_output = run_with_steering(model, tokenizer, alpha * r, coef, p_safe, type='logits', layer=[20])
        loss_add = compute_ce_loss(added_output, t_refusal)

        # kl divergence between post-ablation output and target (harmless)
        log_original = model(p_safe)
        original_output = log_original[torch.arange(log_original.shape[0]), -1, :150000]
        ablated_safe_output = run_with_ablation(model, tokenizer, r.detach(), p_safe, type='logits')
        loss_ret = compute_kl(ablated_safe_output, original_output)

        # total weighted loss
        loss = lambda_abl * loss_abl + lambda_add * loss_add + lambda_ret * loss_ret

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        with torch.no_grad():
            r /= r.norm()

        #print(f"Step {step} | Total Loss: {loss.item():.4f} | Abl: {loss_abl.item():.4f} | Add: {loss_add.item():.4f} | Ret: {loss_ret.item():.4f}")

    return r.detach()


# ---- RCO: Refusal Cone Optimization ----

def sample(B: Float[Tensor, "n d"]) -> Float[Tensor, "d"]:
    """Randomly sample a direction from the cone."""
    n = B.shape[0]
    s = torch.randn(n, device=B.device)
    s = s.abs()
    s = s / s.norm(p=2)
    return torch.matmul(s, B)


def compute_loss(r: Float[Tensor, "d"], compute_fn: Callable, B: Float[Tensor, "n d"]) -> torch.Tensor:
    loss_sample = compute_fn(r)
    loss_basis = torch.stack([compute_fn(bi) for bi in B]).mean()
    return loss_sample + loss_basis


def make_compute_fn(model, tokenizer, batch, alpha=1.0, lambda_abl=1.0, lambda_add=1.0, lambda_ret=1.0):
    """Define function for computing loss based on direction."""
    p_harm, p_safe, t_answer, t_refusal, t_retain = batch

    def compute_fn(r: torch.Tensor) -> torch.Tensor:
        ablated_output = run_with_ablation(model, tokenizer, r, p_harm, type='logits')
        loss_abl = F.cross_entropy(ablated_output, t_answer.detach())

        added_output = run_with_steering(model, tokenizer, alpha * r, p_safe, type='logits', layer=[20])
        loss_add = F.cross_entropy(added_output, t_refusal.detach())

        with torch.no_grad():
            original_logits = model(p_safe)[..., -1, :]
        ablated_safe_output = run_with_ablation(model, tokenizer, r.detach(), p_safe, type='logits')
        loss_ret = F.kl_div(F.log_softmax(ablated_safe_output, dim=-1), F.softmax(original_logits, dim=-1), reduction="batchmean")

        return lambda_abl * loss_abl + lambda_add * loss_add + lambda_ret * loss_ret

    return compute_fn


def make_multi_batch_compute_fn(model, tokenizer, batch_list, alpha=1.0, lambda_abl=1.0, lambda_add=1.0, lambda_ret=1.0):
    """Use for multiple data points."""
    compute_fns = [make_compute_fn(model, tokenizer, batch, alpha, lambda_abl, lambda_add, lambda_ret)
                   for batch in batch_list]

    def compute_fn(r: torch.Tensor) -> torch.Tensor:
        return torch.stack([fn(r) for fn in compute_fns]).mean()

    return compute_fn


def gram_schmidt(B: Float[Tensor, "n d"]) -> Float[Tensor, "n d"]:
    """Orthonormalize basis vectors."""
    n, d = B.shape
    orthonormal_B = []
    for i in range(n):
        vec = B[i]
        for j in range(i):
            proj = (vec @ orthonormal_B[j]) * orthonormal_B[j]
            vec = vec - proj
        vec = vec / vec.norm()
        orthonormal_B.append(vec)
    return torch.stack(orthonormal_B)


def refusal_cone_optimization(
    model, tokenizer, data, d: int, n: int = 5, lr: float = 1e-2, steps: int = 100
) -> Float[Tensor, "n d"]:
    from config import DEVICE

    B = torch.randn(n, d).to(DEVICE)
    B = gram_schmidt(B)
    B.requires_grad_(True)
    opt = torch.optim.Adam([B], lr=lr)

    for i in range(steps):
        batch = random.choice(data)
        compute_fn = make_compute_fn(model, tokenizer, batch)

        B.requires_grad_(True)

        r = sample(B)
        loss = compute_loss(r, compute_fn, B)

        opt.zero_grad()
        loss.backward()
        opt.step()

        with torch.no_grad():
            B = B - lr * B.grad
            B = gram_schmidt(B.detach())

    return B
