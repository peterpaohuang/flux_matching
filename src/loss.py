import torch

HORIZON, SIMULATION_STEPS = 4.0, 4

def flux_matching_loss(f_theta, x, sigma2, q=None, return_t=False):
    B, device = x.shape[0], x.device
    sum_dims = tuple(range(1, x.ndim))

    sigma2 = sigma2.reshape(-1)[0].to(device=device)

    x0 = (x + sigma2.sqrt() * torch.randn_like(x)).detach()
    ref = x.detach().reshape(B, -1)

    def flat(z): return z.reshape(z.shape[0], -1)
    
    def sqdist(a, b):
        return (
            a.square().sum(1, keepdim=True)
            + b.square().sum(1, keepdim=True).t()
            - 2.0 * (a @ b.t())
        )

    def score(z):
        w = torch.softmax(-sqdist(flat(z), ref) / (2.0 * sigma2), dim=-1)
        return ((w @ ref).reshape_as(z) - z) / sigma2

    def exp_langevin_step(z, h):
        rho = torch.exp(-h)
        std = (sigma2 * (1.0 - rho.square()).clamp_min(0.0)).sqrt()
        return z + (1.0 - rho) * sigma2 * score(z) + std * torch.randn_like(z)

    def weights(z0, zt, t):
        rho = torch.exp(-t)
        mu = z0 + (1.0 - rho) * sigma2 * score(z0)
        var = (sigma2 * (1.0 - rho.square()))
        w = torch.softmax(-0.5 * sqdist(flat(mu), flat(zt)) / var, dim=0)
        return w, rho

    def r_theta(z):
        s = score(z)
        u = f_theta(z) - s
        eps = (2 * torch.randint(0, 2, z.shape, device=device) - 1)
        div_u = torch.autograd.grad(
            (u * eps).sum(), z, create_graph=True, retain_graph=True
        )[0]
        return (div_u * eps).sum(dim=sum_dims) + (u * s).sum(dim=sum_dims)

    if q is None:
        t, q_t = torch.rand_like(sigma2) * HORIZON, sigma2.new_tensor(1.0 / HORIZON)
    else:
        t, q_t = [v.squeeze(0) for v in q.sample(sigma2.sqrt()[None])]

    xt = x0
    for _ in range(SIMULATION_STEPS):
        xt = exp_langevin_step(xt, t / SIMULATION_STEPS)

    xt_grad = xt.detach().requires_grad_(True)
    grad_r = torch.autograd.grad(r_theta(xt_grad).sum(), xt_grad)[0].detach()

    w, rho = weights(x0, xt.detach(), t.detach())
    grad_r_hat = ((rho / q_t) * (w @ flat(grad_r))).reshape_as(x0).detach()

    u_theta = f_theta(x0) - score(x0)
    loss = -2.0 * (u_theta * grad_r_hat).sum(dim=sum_dims)

    return (loss, t) if return_t else loss