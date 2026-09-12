"""Sparse Autoencoder (SAE) PyTorch Module with unit-norm constrained decoder."""

from __future__ import annotations
import os
import json
from typing import NamedTuple, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import SAEArchitectureConfig


class SAEOuput(NamedTuple):
    sae_out: torch.Tensor          # Reconstructed activations [batch, d_model]
    feature_acts: torch.Tensor     # Sparse latent activations [batch, d_sae]
    loss: torch.Tensor             # Total loss = l2_loss + l1_loss
    l2_loss: torch.Tensor          # Mean squared reconstruction error
    l1_loss: torch.Tensor          # Sparsity penalty


class SparseAutoencoder(nn.Module):
    """Anthropic / Cunningham et al. formulation Sparse Autoencoder.
    
    Architecture:
        f(x) = ReLU(W_enc (x - b_dec) + b_enc)
        x_hat = W_dec f(x) + b_dec

    The decoder weights are constrained to have unit L2 norm per feature column,
    preventing the model from trivializing the L1 sparsity penalty.
    """

    def __init__(self, config: SAEArchitectureConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.d_sae = config.d_sae
        self.l1_coefficient = config.l1_coefficient

        # Encoder: maps (x - b_dec) -> latents
        self.W_enc = nn.Linear(self.d_model, self.d_sae, bias=True)
        # Decoder: maps latents -> reconstructed x (no bias; b_dec is handled explicitly)
        self.W_dec = nn.Linear(self.d_sae, self.d_model, bias=False)
        # Geometric centering bias (b_dec in Anthropic formulation)
        self.b_dec = nn.Parameter(torch.zeros(self.d_model))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initializes weights following best practices:
        - Decoder columns initialized to random unit vectors or Kaiming uniform.
        - Encoder initialized as transpose of decoder.
        - Biases initialized to zero.
        """
        nn.init.kaiming_uniform_(self.W_dec.weight, nonlinearity="linear")
        self.set_decoder_norm_to_unit()

        # Initialize encoder as transpose of decoder
        with torch.no_grad():
            self.W_enc.weight.data.copy_(self.W_dec.weight.data.t())
            nn.init.zeros_(self.W_enc.bias)
            nn.init.zeros_(self.b_dec)

    @torch.no_grad()
    def set_decoder_norm_to_unit(self) -> None:
        """Projects each feature column of the decoder weight matrix to unit L2 norm.
        
        W_dec.weight has shape (d_model, d_sae).
        Column j corresponds to feature j in d_sae.
        """
        col_norms = torch.norm(self.W_dec.weight.data, p=2, dim=0, keepdim=True)
        self.W_dec.weight.data.div_(col_norms.clamp_min(1e-8))

    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self) -> None:
        """Optional projection to remove gradient components parallel to decoder directions,
        preventing step overshoot on the unit sphere.
        """
        if self.W_dec.weight.grad is None:
            return
        # W_dec.weight: (d_model, d_sae)
        w = self.W_dec.weight.data
        grad = self.W_dec.weight.grad.data
        # Dot product along d_model dimension (dim 0)
        proj = (grad * w).sum(dim=0, keepdim=True)  # shape (1, d_sae)
        self.W_dec.weight.grad.data.sub_(proj * w)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes residual stream activations into sparse latent space.
        
        Args:
            x: Tensor of shape (..., d_model)
        Returns:
            feature_acts: Tensor of shape (..., d_sae) with non-negative activations
        """
        # Center by subtracting decoder bias
        x_centered = x - self.b_dec
        # Forward through encoder with ReLU activation
        latents = self.W_enc(x_centered)
        return F.relu(latents)

    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        """Decodes sparse latents back to residual stream representation.
        
        Args:
            feature_acts: Tensor of shape (..., d_sae)
        Returns:
            x_hat: Reconstructed activation of shape (..., d_model)
        """
        return self.W_dec(feature_acts) + self.b_dec

    def forward(
        self,
        x: torch.Tensor,
        l1_override: Optional[float] = None,
    ) -> SAEOuput:
        """Forward pass with loss computation.
        
        Args:
            x: Input activation batch of shape [batch_size, d_model]
            l1_override: Optional override for L1 sparsity penalty coefficient
        """
        feature_acts = self.encode(x)
        sae_out = self.decode(feature_acts)

        # L2 reconstruction loss per token, averaged over batch
        # shape: sum over d_model, mean over batch
        l2_loss = (x - sae_out).pow(2).sum(dim=-1).mean()

        # L1 sparsity penalty: sum of absolute activations per token
        l1_coeff = l1_override if l1_override is not None else self.l1_coefficient
        l1_loss = l1_coeff * feature_acts.sum(dim=-1).mean()

        total_loss = l2_loss + l1_loss

        return SAEOuput(
            sae_out=sae_out,
            feature_acts=feature_acts,
            loss=total_loss,
            l2_loss=l2_loss,
            l1_loss=l1_loss,
        )

    def get_feature_direction(self, feature_idx: int) -> torch.Tensor:
        """Returns the unit-norm decoder direction vector for a given latent feature.
        
        Args:
            feature_idx: index in [0, d_sae - 1]
        Returns:
            direction: 1D Tensor of shape (d_model,)
        """
        assert 0 <= feature_idx < self.d_sae, f"Invalid feature index {feature_idx}"
        return self.W_dec.weight[:, feature_idx].detach()

    @staticmethod
    def compute_l0(feature_acts: torch.Tensor) -> float:
        """Calculates average number of active features per token (L0 norm)."""
        active_counts = (feature_acts > 0).float().sum(dim=-1)
        return active_counts.mean().item()

    @staticmethod
    def compute_fve(x: torch.Tensor, sae_out: torch.Tensor) -> float:
        """Calculates Fraction of Variance Explained (FVE) on a batch.
        
        FVE = 1 - Var(x - x_hat) / Var(x)
        """
        with torch.no_grad():
            residual_var = (x - sae_out).var(dim=0).sum()
            total_var = x.var(dim=0).sum()
            if total_var.item() == 0:
                return 0.0
            fve = 1.0 - (residual_var / total_var).item()
            return float(fve)

    def save_pretrained(self, save_dir: str) -> None:
        """Saves SAE weights and architecture config."""
        os.makedirs(save_dir, exist_ok=True)
        # Save config
        config_path = os.path.join(save_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config.__dict__, f, indent=2)
        # Save weights
        weights_path = os.path.join(save_dir, "sae_weights.pt")
        torch.save(self.state_dict(), weights_path)

    @classmethod
    def from_pretrained(cls, load_dir: str, device: Optional[torch.device] = None) -> SparseAutoencoder:
        """Loads SAE from saved directory."""
        config_path = os.path.join(load_dir, "config.json")
        with open(config_path, "r") as f:
            cfg_dict = json.load(f)
        config = SAEArchitectureConfig(**cfg_dict)
        sae = cls(config)
        weights_path = os.path.join(load_dir, "sae_weights.pt")
        map_location = device if device is not None else "cpu"
        sae.load_state_dict(torch.load(weights_path, map_location=map_location))
        if device is not None:
            sae.to(device)
        sae.eval()
        return sae
