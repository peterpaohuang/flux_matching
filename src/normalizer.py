import torch

class LossNormalizer(torch.nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.input_layer = torch.nn.Linear(1, hidden_dim)
        self.output_layer = torch.nn.Linear(hidden_dim, 1)

    def forward(self, sigma):
        """
        Args:
            sigma: [B] noise standard deviations (positive floats)
        Returns:
            [B] learned per-sigma scale factors (positive, via softplus)
        """
        sigma = sigma.log() / 4
        return self.output_layer(self.input_layer(sigma.reshape(-1, 1))).squeeze(-1)