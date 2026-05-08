import torch
import torch.nn as nn
import torch.nn.functional as F

HORIZON = 4.0

class ExponentialIS(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.input_layer = nn.Linear(1, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def _lambdas(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sigma: [B] noise standard deviations (positive floats)
        Returns:
            [B] positive rate parameters for the truncated exponential
        """
        sigma = sigma.log() / 4
        return 1.0 + F.softplus(self.output_layer(self.input_layer(sigma.reshape(-1, 1))).squeeze(-1))

    def sample(self, sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample t ~ TruncExp(lambda(sigma), [0, HORIZON]).

        Args:
            sigma: [B] noise standard deviations (positive floats)
        Returns:
            t:       [B] sampled times in [0, HORIZON]
            density: [B] q(t) evaluated at the sampled t
        """
        with torch.no_grad():
            lam = self._lambdas(sigma)
        B = lam.shape[0]
        device, dtype = lam.device, lam.dtype

        u = torch.rand(1, device=device, dtype=dtype).expand(B)
        one_minus_exp = -torch.expm1(-lam * HORIZON)
        t = -torch.log1p(-u * one_minus_exp) / lam
        t = t.clamp(0.0, HORIZON)

        density = self.log_density(t, sigma).exp()
        return t, density

    def log_density(self, t: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Log-density of TruncExp(lambda(sigma), [0, HORIZON]):
            log q(t) = log(lambda) - lambda*t - log(1 - exp(-lambda*H))

        Args:
            t:     [B] times in [0, HORIZON]
            sigma: [B] noise standard deviations (positive floats)
        Returns:
            [B] log q(t)
        """
        lam = self._lambdas(sigma)
        t = t.reshape(-1).to(lam)
        log_norm = torch.log(-torch.expm1(-lam * HORIZON))
        return torch.log(lam) - lam * t - log_norm

    def reinforce_loss(
        self,
        t: torch.Tensor,
        sigma: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        REINFORCE loss for updating the proposal distribution.

        Args:
            t:      [B] sampled times (detached)
            sigma:  [B] noise standard deviations (positive floats)
            target: [B] IS-weight-normalized reward signals (detached)
        Returns:
            [B] per-sample losses; gradients flow through log_density only
        """
        t = t.detach().reshape(-1)
        target = target.detach().reshape(-1)

        log_q_live = self.log_density(t, sigma)
        return -target * log_q_live
