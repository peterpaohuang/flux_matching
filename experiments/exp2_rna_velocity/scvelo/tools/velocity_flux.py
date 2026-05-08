import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from abc import abstractmethod
from scvelo.dataloader.dataloader import VeloDataLoader
from anndata import AnnData
from scvelo import logging as logg

def flux_matching_loss(f, x, sigma2):
    """
    Gene-local Flux Matching loss for DynamicalKineticVectorField.

    Args:
        f: DynamicalKineticVectorField instance (unwrapped from DataParallel)
        x: [B, G, 2] tensor of (u, s) pairs
        sigma2: scalar tensor, noise variance

    Returns:
        scalar loss
    """
    eps = 1e-12
    B, G, _ = x.shape
    device, dtype = x.device, x.dtype

    K, M = 4, 4
    horizon = float(K)

    sigma2 = sigma2.clamp_min(eps)
    sigma = torch.sqrt(sigma2)

    # Gene-local score: [B, G, 2] -> [B, G, 2], kernel over B cells per gene
    def score_fn(z):
        ref = x.detach().permute(1, 0, 2)       # [G, B, 2]
        q = z.permute(1, 0, 2)                  # [G, B, 2]
        q2 = (q * q).sum(-1, keepdim=True)      # [G, B, 1]
        r2 = (ref * ref).sum(-1).unsqueeze(1)   # [G, 1, B]
        cross = torch.einsum("gid,gjd->gij", q, ref)
        dist2 = (q2 + r2 - 2.0 * cross).clamp_min(0.0)
        w = torch.softmax(-dist2 / (2.0 * sigma2), dim=-1)
        mean_ref = torch.einsum("gij,gjd->gid", w, ref)
        return ((mean_ref - q) / sigma2).permute(1, 0, 2)  # [B, G, 2]

    def _param_view(p):
        return p.view(1, G)

    def _field(u, s, detach_params=False):
        beta = F.softplus(f.beta_unconstrained) + f.min_rate
        gamma = F.softplus(f.gamma_unconstrained) + f.min_rate
        a_u = f.transcription_u
        a_s = f.transcription_s
        if detach_params:
            beta, gamma, a_u, a_s = beta.detach(), gamma.detach(), a_u.detach(), a_s.detach()
        preact = _param_view(a_u) * u + _param_view(a_s) * s
        if f.use_bias:
            bias = f.transcription_bias if not detach_params else f.transcription_bias.detach()
            preact = preact + _param_view(bias)
        drive = F.softplus(preact) + f.min_transcription
        du = drive - _param_view(beta) * u
        ds = _param_view(beta) * u - _param_view(gamma) * s
        return torch.stack([du, ds], dim=-1), drive, preact  # [B, G, 2]

    tau = torch.rand((), device=device, dtype=dtype) * horizon

    # Anchor: perturb data
    x_anchor = x + sigma * torch.randn_like(x)

    # OU rollout to x_tau
    x_t = x_anchor.detach()
    tau_step = tau / float(M)
    rho_step = torch.exp(-tau_step)
    var_step = (sigma2 * (1.0 - rho_step ** 2)).clamp_min(eps)
    for _ in range(M):
        s_t = score_fn(x_t)
        mu = x_t + sigma2 * s_t
        x_t = rho_step * x_t + (1.0 - rho_step) * mu + torch.sqrt(var_step) * torch.randn_like(x_t)
    x_tau = x_t.detach()

    # Endpoint: compute r_tau and its gradient wrt x_tau
    x_tau_g = x_tau.requires_grad_(True)
    score_tau = score_fn(x_tau_g)
    f_tau, _, _ = _field(x_tau_g[..., 0], x_tau_g[..., 1], detach_params=True)
    mismatch = f_tau - score_tau

    eps_h = (torch.randint(0, 2, x_tau_g.shape, device=device) * 2 - 1).to(dtype)
    proj = (mismatch * eps_h).sum(dim=(1, 2))
    div_mismatch = (torch.autograd.grad(proj.sum(), x_tau_g, create_graph=True, retain_graph=True)[0] * eps_h).sum(dim=(1, 2))
    r_tau = div_mismatch + (mismatch * score_tau).sum(dim=(1, 2))
    grad_r_tau = torch.autograd.grad(r_tau.sum(), x_tau_g, create_graph=False)[0].detach()  # [B, G, 2]

    # Responsibilities [G, B_anchor, B_tau]
    rho = torch.exp(-tau)
    var_ou = (sigma2 * (1.0 - rho ** 2)).clamp_min(eps)
    s_anchor = score_fn(x_anchor.detach())
    means = (x_anchor.detach() + (1.0 - rho) * sigma2 * s_anchor).permute(1, 0, 2)  # [G, B, 2]
    ends = x_tau.permute(1, 0, 2)                                                     # [G, B, 2]
    m2 = (means * means).sum(-1, keepdim=True)
    e2 = (ends * ends).sum(-1).unsqueeze(1)
    dist2 = (m2 + e2 - 2.0 * torch.einsum("gid,gjd->gij", means, ends)).clamp_min(0.0)
    resp = torch.softmax(-0.5 * dist2 / var_ou, dim=1)  # [G, B, B]

    q_hat = (rho / (1.0 / horizon) * torch.einsum(
        "gij,gjd->gid", resp, grad_r_tau.permute(1, 0, 2)
    )).permute(1, 0, 2).detach()  # [B, G, 2]

    # Main branch
    f0, drive0, preact0 = _field(x_anchor[..., 0], x_anchor[..., 1], detach_params=False)
    mismatch0 = f0 - score_fn(x_anchor)
    loss = (-2.0 * (mismatch0 * q_hat).sum(dim=(1, 2)) * sigma2 ** 2).mean()

    # Update caches for eval
    f._last_beta = (F.softplus(f.beta_unconstrained) + f.min_rate).detach()
    f._last_gamma = (F.softplus(f.gamma_unconstrained) + f.min_rate).detach()
    f._last_transcription_u = f.transcription_u.detach()
    f._last_transcription_s = f.transcription_s.detach()
    f._last_transcription_bias = None if f.transcription_bias is None else f.transcription_bias.detach()
    f._last_transcription_preact = preact0.detach()
    f._last_transcription_drive = drive0.detach()

    return loss

class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, model, optimizer):
        self.device = 'cuda'
        self.model = model.to(self.device)
        self.model = torch.nn.DataParallel(model, device_ids=[0])

        self.optimizer = optimizer

        self.start_epoch = 1
        self.epochs = 100

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self, callback=None, callback_freq=1):
        """
        Full training logic
        """
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._train_epoch(epoch)

    def train_with_epoch_callback(self, callback, freq):
        self.train(callback, freq)

    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning(
                "Warning: There's no GPU available on this machine,"
                "training will be performed on CPU."
            )
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning(
                "Warning: The number of GPU's configured to use is {}, but only {} are available "
                "on this machine.".format(n_gpu_use, n_gpu)
            )
            n_gpu_use = n_gpu
        device = torch.device("cuda:0" if n_gpu_use > 0 else "cpu")
        list_ids = list(range(n_gpu_use))
        return device, list_ids


class FluxTrainer(BaseTrainer):
    def __init__(self, model, optimizer, dataloader):
        super().__init__(model, optimizer)
        self.dataloader = dataloader
        self.len_epoch = len(self.dataloader)

    def _train_epoch(self, epoch):
        self.model.train()

        # IMPORTANT: use the normal minibatched dataloader, not large_batch(...)
        loader = self.dataloader

        for batch_idx, batch_data in enumerate(loader):
            self.optimizer.zero_grad(set_to_none=True)

            batch_data = {
                k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v
                for k, v in batch_data.items()
            }

            # Unwrap DataParallel because the loss uses autograd.grad internally
            f = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model

            x_u = batch_data["Ux_sz"]
            x_s = batch_data["Sx_sz"]
            x = torch.stack([x_u, x_s], dim=-1)  # [B, G, 2]
            sigma2 = torch.tensor(1e-3, device=x.device, dtype=x.dtype)

            loss = flux_matching_loss(f, x, sigma2)

            loss.backward()
            self.optimizer.step()

    def eval(self, eval_loader, return_kinetic_rates=False, n_genes=2000):
        self.model.eval()
        n_cells = len(eval_loader.dataset)

        velo_mat = np.zeros((n_cells, n_genes), dtype=np.float32)
        velo_mat_u = np.zeros((n_cells, n_genes), dtype=np.float32)
        kinetic_rates = {}

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(eval_loader):
                idx = batch_data["idx"].cpu().numpy()

                x_u = batch_data["Ux_sz"].to(self.device, non_blocking=True)
                x_s = batch_data["Sx_sz"].to(self.device, non_blocking=True)
                output = self.model(x_u, x_s)
                pred_u = output[:, 0:n_genes].cpu().numpy()
                pred_s = output[:, n_genes:2 * n_genes].cpu().numpy()

                velo_mat[idx] = pred_s
                velo_mat_u[idx] = pred_u

                model_ref = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
                cur_kinetic_rates = model_ref.get_current_batch_kinetic_rates()

                for k, v in cur_kinetic_rates.items():
                    if v is None:
                        continue

                    vv = v.detach().cpu().numpy()

                    # Gene-wise quantities: keep one copy only
                    if vv.ndim == 1:
                        if k not in kinetic_rates:
                            kinetic_rates[k] = vv.copy()
                        continue

                    # Cell-specific quantities: scatter back by cell index
                    if vv.ndim == 2 and vv.shape[0] == len(idx):
                        if k not in kinetic_rates:
                            kinetic_rates[k] = np.zeros((n_cells, vv.shape[1]), dtype=np.float32)
                        kinetic_rates[k][idx] = vv
                        continue

                    raise ValueError(f"Unexpected shape for kinetic rate '{k}': {vv.shape}")

        return velo_mat, velo_mat_u, kinetic_rates


class DynamicalKineticVectorField(nn.Module):
    """
    Smooth mechanistic family:

        f_g(u_g, s_g) = softplus(a_g * u_g + b_g * s_g + c_g)
        du_g = f_g(u_g, s_g) - beta_g * u_g
        ds_g = beta_g * u_g - gamma_g * s_g

    Notes:
    - beta_g and gamma_g are positive.
    - f_g(u_g, s_g) is a positive, smooth, gene-local transcription drive.
    - This removes the hard/soft on-off boundary logic entirely.
    - With a_g = b_g = c_g = 0 at initialization, the transcription drive starts
      as a constant softplus(0), so the model begins as a simple linear kinetic ODE.
    """

    def __init__(
        self,
        n_genes: int,
        min_rate: float = 1e-6,
        min_transcription: float = 1e-6,
        use_bias: bool = True,
    ):
        super().__init__()
        G = n_genes
        self.n_genes = G
        self.min_rate = min_rate
        self.min_transcription = min_transcription
        self.use_bias = use_bias

        # Positive kinetic parameters
        self.beta_unconstrained = nn.Parameter(torch.zeros(G))
        self.gamma_unconstrained = nn.Parameter(torch.zeros(G))

        # Signed transcription-drive coefficients
        self.transcription_u = nn.Parameter(torch.zeros(G))
        self.transcription_s = nn.Parameter(torch.zeros(G))

        if self.use_bias:
            self.transcription_bias = nn.Parameter(torch.zeros(G))
        else:
            self.register_parameter("transcription_bias", None)

        self._last_beta = None
        self._last_gamma = None
        self._last_transcription_u = None
        self._last_transcription_s = None
        self._last_transcription_bias = None
        self._last_transcription_preact = None
        self._last_transcription_drive = None

    def _param_view(self, p: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return p.view(*([1] * (x.ndim - 1)), self.n_genes)

    def forward(self, x_u: torch.Tensor, x_s: torch.Tensor) -> torch.Tensor:
        if x_u.shape != x_s.shape:
            raise ValueError(f"x_u and x_s must have the same shape, got {x_u.shape} vs {x_s.shape}")
        if x_u.shape[-1] != self.n_genes:
            raise ValueError(
                f"Last dimension must equal n_genes={self.n_genes}, got {x_u.shape[-1]}"
            )

        beta = F.softplus(self.beta_unconstrained) + self.min_rate
        gamma = F.softplus(self.gamma_unconstrained) + self.min_rate

        a_u = self.transcription_u
        a_s = self.transcription_s

        beta_b = self._param_view(beta, x_u)
        gamma_b = self._param_view(gamma, x_u)
        a_u_b = self._param_view(a_u, x_u)
        a_s_b = self._param_view(a_s, x_u)

        transcription_preact = a_u_b * x_u + a_s_b * x_s
        if self.use_bias:
            transcription_preact = transcription_preact + self._param_view(self.transcription_bias, x_u)

        transcription_drive = F.softplus(transcription_preact) + self.min_transcription

        du = transcription_drive - beta_b * x_u
        ds = beta_b * x_u - gamma_b * x_s

        self._last_beta = beta.detach()
        self._last_gamma = gamma.detach()
        self._last_transcription_u = a_u.detach()
        self._last_transcription_s = a_s.detach()
        self._last_transcription_bias = None if self.transcription_bias is None else self.transcription_bias.detach()
        self._last_transcription_preact = transcription_preact.detach()
        self._last_transcription_drive = transcription_drive.detach()

        return torch.cat([du, ds], dim=-1)

    def get_current_batch_kinetic_rates(self):
        return {
            "beta": self._last_beta,
            "gamma": self._last_gamma,
            "transcription_u": self._last_transcription_u,
            "transcription_s": self._last_transcription_s,
            "transcription_bias": self._last_transcription_bias,
            "transcription_preact": self._last_transcription_preact,
            "transcription_drive": self._last_transcription_drive,
        }


def flux_velocity(
    adata: AnnData,
    model_family: str = "dynamical",
    model_kwargs: dict | None = None,
    lr: float = 1e-3,
    epochs: int = 100,
):
    if model_kwargs is None:
        model_kwargs = {}

    n_cells, n_genes = adata.layers["Ms"].shape

    train_loader = VeloDataLoader(data_source=adata, shuffle=True)
    eval_loader = VeloDataLoader(data_source=adata, shuffle=False)

    if model_family == "dynamical":
        model = DynamicalKineticVectorField(n_genes=n_genes, **model_kwargs)
    else:
        raise ValueError(
            f"Unknown model_family={model_family}. "
            f"Expected one of: steady_state, dynamical, full_state_dependent."
        )

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=lr)

    trainer = FluxTrainer(model, optimizer, train_loader)
    trainer.epochs = epochs
    trainer.train()

    velo_mat, velo_mat_u, kinetic_rates = trainer.eval(
        eval_loader,
        return_kinetic_rates=True,
        n_genes=n_genes,
    )

    assert adata.layers["Ms"].shape == velo_mat.shape
    adata.layers["velocity"] = velo_mat
    adata.layers["velocity_unspliced"] = velo_mat_u

    logg.hint("added 'velocity' (adata.layers)")
    logg.hint("added 'velocity_unspliced' (adata.layers)")

    for k, v in kinetic_rates.items():
        if v is None:
            continue

        if v.ndim == 1:
            if len(v) != n_genes:
                raise ValueError(f"Gene-wise quantity '{k}' has length {len(v)}, expected {n_genes}")
            adata.var[k] = v
            logg.hint(f"added '{k}' (adata.var)")

        elif v.ndim == 2:
            if v.shape != (n_cells, n_genes):
                raise ValueError(
                    f"Cell-specific quantity '{k}' has shape {v.shape}, expected {(n_cells, n_genes)}"
                )
            adata.layers["cell_specific_" + k] = v
            logg.hint(f"added 'cell_specific_{k}' (adata.layers)")

        else:
            raise ValueError(f"Unexpected ndim for '{k}': {v.ndim}")

    adata.uns["flux_model_family"] = model_family