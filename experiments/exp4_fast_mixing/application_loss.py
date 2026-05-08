import torch
import math

def _sinkhorn_cost_matrix_edm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_flat = x.reshape(x.shape[0], -1)
    y_flat = y.reshape(y.shape[0], -1)
    x2 = (x_flat * x_flat).sum(dim=1, keepdim=True)
    y2 = (y_flat * y_flat).sum(dim=1, keepdim=True)
    return x2 + y2.t() - 2.0 * (x_flat @ y_flat.t())


def _sinkhorn_ot_cost_edm(C: torch.Tensor, reg: float, num_iters: int = 50) -> torch.Tensor:
    B = C.shape[0]
    log_a = -math.log(B) * torch.ones(B, device=C.device, dtype=C.dtype)
    log_b = log_a.clone()
    log_K = -C / reg
    f = torch.zeros(B, device=C.device, dtype=C.dtype)
    g = torch.zeros(B, device=C.device, dtype=C.dtype)
    for _ in range(num_iters):
        f = log_a - torch.logsumexp(log_K + g.unsqueeze(0), dim=1)
        g = log_b - torch.logsumexp(log_K + f.unsqueeze(1), dim=0)
    log_P = f.unsqueeze(1) + log_K + g.unsqueeze(0)
    P = torch.exp(log_P)
    return (P * C).sum()


def _sinkhorn_divergence_edm(
    x: torch.Tensor,
    y: torch.Tensor,
    reg: float = 0.05,
    num_iters: int = 50,
) -> torch.Tensor:
    Cxy = _sinkhorn_cost_matrix_edm(x, y)
    Cxx = _sinkhorn_cost_matrix_edm(x, x)
    Cyy = _sinkhorn_cost_matrix_edm(y, y)
    wxy = _sinkhorn_ot_cost_edm(Cxy, reg=reg, num_iters=num_iters)
    wxx = _sinkhorn_ot_cost_edm(Cxx, reg=reg, num_iters=num_iters)
    wyy = _sinkhorn_ot_cost_edm(Cyy, reg=reg, num_iters=num_iters)
    return (wxy - 0.5 * wxx - 0.5 * wyy).clamp(min=0.0)


def _adaptive_langevin_step_edm(
    x: torch.Tensor,
    score: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    eps = (2.0 * sigma ** 2)[:, None, None, None] * 0.1
    z = torch.randn_like(x)
    return x + eps * score + (2.0 * eps).sqrt() * z


def mixing_loss_fixed_sigma_edm(
    f_theta,
    clean_batch: torch.Tensor,
    sigmas: torch.Tensor
) -> tuple:
    """
    Normalized mixing loss at fixed sigma (EDM x0-predictor edition):

        L_rel = SD_eps(X^(1), Y) / (SD_eps(X^(0), Y) + tau)

    Score is derived from the x0-predictor as:
        score(x, sigma) = (D_x(x, sigma) - x) / sigma^2

    One adaptive Langevin corrector step (Algorithm 4, Song et al. 2021) is used.
    """
    B = clean_batch.shape[0]
    half = B // 2
    device, dtype = clean_batch.device, clean_batch.dtype

    sigma_val = sigmas.reshape(-1)[0]
    sigma2_img = (sigma_val ** 2).view(1, *([1] * (clean_batch.ndim - 1)))

    clean_y, clean_x = clean_batch[:half], clean_batch[half:half*2]

    y = clean_y + sigma_val * torch.randn_like(clean_y)

    x_stationary = clean_x + sigma_val * torch.randn_like(clean_x)
    x0 = x_stationary + sigma_val * torch.randn_like(x_stationary)

    sigma_batch = sigma_val.expand(half)

    # One adaptive Langevin step with gradients for backprop through drift
    drift = f_theta(x0, sigma_batch[:, None, None, None])
    xm = _adaptive_langevin_step_edm(x0, drift, sigma_batch)

    d_m = _sinkhorn_divergence_edm(xm, y.detach())

    with torch.no_grad():
        d_0 = _sinkhorn_divergence_edm(x0.detach(), y.detach())

    loss = d_m / (d_0 + 1e-3)

    return loss