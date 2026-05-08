import torch


def dsm_loss_x0(D_theta, x, sigma2):
    B = x.shape[0]
    sigma = sigma2.sqrt()

    sigma_x = sigma.reshape(B, *([1] * (x.ndim - 1)))
    x_noised = (x + sigma_x * torch.randn_like(x)).detach()

    # Stable x0 target: E[x | x_noised] under the batch KDE.
    ps = x_noised.reshape(B, -1)
    rv = x.reshape(B, -1)

    sig = sigma.reshape(B, 1)
    denom = 2.0 * sig.square()

    x2 = (ps * ps).sum(dim=1, keepdim=True)
    y2 = (rv * rv).sum(dim=1, keepdim=True).t()
    xy = ps @ rv.t()
    dist2 = x2 + y2 - 2.0 * xy

    log_weights = -dist2 / denom
    log_weights = log_weights - log_weights.max(dim=1, keepdim=True).values
    weights = torch.exp(log_weights)
    weights = weights / weights.sum(dim=1, keepdim=True)

    target_x0 = weights @ rv

    weight = (sigma.square() + 0.5 ** 2) / (sigma * 0.5).square()
    loss_4shape = weight.reshape(B, 1) * (
        target_x0 - D_theta(x_noised, sigma).flatten(1)
    ).square()

    return loss_4shape.mean(dim=-1)


def dsm_loss_epsilon(f_theta, x, sigma2):
    B = x.shape[0]
    sigma = sigma2.sqrt()

    sigma_x = sigma.reshape(B, *([1] * (x.ndim - 1)))
    eps = torch.randn_like(x)
    x_noised = (x + sigma_x * eps).detach()

    # Stable epsilon target:
    # eps = (x_noised - E[x | x_noised]) / sigma
    ps = x_noised.reshape(B, -1)
    rv = x.reshape(B, -1)

    sig = sigma.reshape(B, 1)
    denom = 2.0 * sig.square()

    x2 = (ps * ps).sum(dim=1, keepdim=True)
    y2 = (rv * rv).sum(dim=1, keepdim=True).t()
    xy = ps @ rv.t()
    dist2 = x2 + y2 - 2.0 * xy

    log_weights = -dist2 / denom
    log_weights = log_weights - log_weights.max(dim=1, keepdim=True).values
    weights = torch.exp(log_weights)
    weights = weights / weights.sum(dim=1, keepdim=True)

    target_x0 = weights @ rv
    target_eps = (ps - target_x0) / sig

    loss_4shape = (
        f_theta(x_noised, sigma2).flatten(1) + target_eps.flatten(1)
    ).square()

    return loss_4shape.mean(dim=-1)