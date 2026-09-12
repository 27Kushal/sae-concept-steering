"""Configuration definitions for model, data collection, SAE training, and steering."""

from __future__ import annotations
import os
import json
from dataclasses import dataclass, field, asdict
from typing import Optional, List
import torch


def get_device(preferred: Optional[str] = None) -> torch.device:
    """Detects available device prioritizing CUDA > MPS > CPU with fallback safety.
    
    Local runs on Apple Silicon use MPS if available, falling back to CPU for
    operations not supported by MPS.
    """
    if preferred is not None:
        return torch.device(preferred)

    if torch.cuda.is_available():
        return torch.device("cuda")

    # On Apple Silicon (macOS), check MPS
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # Set fallback env variables if not already set
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        os.environ.setdefault("TRANSFORMERLENS_ALLOW_MPS", "1")
        return torch.device("mps")

    return torch.device("cpu")


@dataclass
class ModelConfig:
    """Base LLM configuration."""
    model_name: str = "gpt2"
    d_model: int = 768
    target_layer: int = 6
    hook_name: str = "blocks.6.hook_resid_post"
    device: str = "auto"

    def resolved_device(self) -> torch.device:
        pref = None if self.device == "auto" else self.device
        return get_device(pref)


@dataclass
class SAEArchitectureConfig:
    """Sparse Autoencoder architecture hyperparameters."""
    d_model: int = 768
    expansion_factor: int = 8  # d_sae = expansion_factor * d_model
    d_sae: int = 6144
    l1_coefficient: float = 3e-4
    normalize_decoder: bool = True
    init_decoder_norm: float = 1.0

    def __post_init__(self):
        if self.d_sae != self.expansion_factor * self.d_model:
            self.d_sae = self.expansion_factor * self.d_model


@dataclass
class DataCollectionConfig:
    """Settings for collecting and sharding residual stream activations."""
    dataset_name: str = "NeelNanda/pile-10k"
    dataset_config: Optional[str] = None
    split: str = "train"
    context_length: int = 256
    batch_size: int = 16
    total_tokens: int = 100_000  # local small: 10k-100k, colab full: 10M-50M
    shard_size_tokens: int = 50_000
    cache_dir: str = "data/activations"
    seed: int = 42


@dataclass
class TrainingConfig:
    """SAE training optimization settings."""
    lr: float = 4e-4
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    batch_size: int = 2048  # batch of activation vectors
    total_steps: int = 2000
    warmup_steps: int = 200
    eval_every: int = 100
    save_every: int = 500
    checkpoint_dir: str = "checkpoints"
    dead_feature_threshold: float = 1e-6
    dead_feature_window: int = 10_000  # tokens over which to track dead features


@dataclass
class SteeringConfig:
    """Settings for activation steering and evaluation."""
    target_layer: int = 6
    hook_name: str = "blocks.6.hook_resid_post"
    steering_alphas: List[float] = field(
        default_factory=lambda: [-6.0, -3.0, 0.0, 3.0, 6.0, 9.0]
    )
    max_new_tokens: int = 40
    temperature: float = 0.7
    top_p: float = 0.9
    num_samples_per_prompt: int = 3
    results_dir: str = "results"


@dataclass
class ProjectConfig:
    """Unified configuration with preset profiles ('small' for local, 'full' for Colab)."""
    scale: str = "small"
    model: ModelConfig = field(default_factory=ModelConfig)
    sae: SAEArchitectureConfig = field(default_factory=SAEArchitectureConfig)
    data: DataCollectionConfig = field(default_factory=DataCollectionConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    steering: SteeringConfig = field(default_factory=SteeringConfig)

    @classmethod
    def create(cls, scale: str = "small") -> ProjectConfig:
        """Instantiate preconfigured settings for 'small' (local M4) or 'full' (Colab T4)."""
        if scale == "small":
            # Lightweight config tailored for local testing & fast dry-runs
            return cls(
                scale="small",
                model=ModelConfig(
                    model_name="gpt2",
                    d_model=768,
                    target_layer=6,
                    hook_name="blocks.6.hook_resid_post",
                    device="auto",
                ),
                sae=SAEArchitectureConfig(
                    d_model=768,
                    expansion_factor=8,
                    d_sae=6144,
                    l1_coefficient=3e-4,
                ),
                data=DataCollectionConfig(
                    dataset_name="NeelNanda/pile-10k",
                    dataset_config=None,
                    split="train",
                    context_length=128,
                    batch_size=8,
                    total_tokens=20_000,
                    shard_size_tokens=10_000,
                    cache_dir="data/activations_small",
                ),
                train=TrainingConfig(
                    lr=4e-4,
                    batch_size=512,
                    total_steps=200,
                    warmup_steps=20,
                    eval_every=50,
                    save_every=100,
                    checkpoint_dir="checkpoints/small",
                ),
                steering=SteeringConfig(
                    steering_alphas=[-4.0, 0.0, 4.0],
                    max_new_tokens=30,
                    num_samples_per_prompt=2,
                    results_dir="results/small",
                ),
            )
        elif scale == "full":
            # Scaled config tailored for Google Colab with GPU acceleration
            return cls(
                scale="full",
                model=ModelConfig(
                    model_name="gpt2",
                    d_model=768,
                    target_layer=6,
                    hook_name="blocks.6.hook_resid_post",
                    device="auto",
                ),
                sae=SAEArchitectureConfig(
                    d_model=768,
                    expansion_factor=8,
                    d_sae=6144,
                    l1_coefficient=3e-4,
                ),
                data=DataCollectionConfig(
                    dataset_name="NeelNanda/pile-10k",
                    dataset_config=None,
                    split="train",
                    context_length=256,
                    batch_size=32,
                    total_tokens=10_000_000,
                    shard_size_tokens=500_000,
                    cache_dir="data/activations_full",
                ),
                train=TrainingConfig(
                    lr=4e-4,
                    batch_size=4096,
                    total_steps=10_000,
                    warmup_steps=1000,
                    eval_every=500,
                    save_every=2000,
                    checkpoint_dir="checkpoints/full",
                ),
                steering=SteeringConfig(
                    steering_alphas=[-8.0, -4.0, 0.0, 4.0, 8.0, 12.0],
                    max_new_tokens=50,
                    num_samples_per_prompt=4,
                    results_dir="results/full",
                ),
            )
        else:
            raise ValueError(f"Unknown scale: {scale}. Must be 'small' or 'full'.")

    def to_dict(self) -> dict:
        return asdict(self)

    def save_json(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, filepath: str) -> ProjectConfig:
        with open(filepath, "r") as f:
            data = json.load(f)
        cfg = cls()
        cfg.scale = data.get("scale", "small")
        cfg.model = ModelConfig(**data.get("model", {}))
        cfg.sae = SAEArchitectureConfig(**data.get("sae", {}))
        cfg.data = DataCollectionConfig(**data.get("data", {}))
        cfg.train = TrainingConfig(**data.get("train", {}))
        cfg.steering = SteeringConfig(**data.get("steering", {}))
        return cfg
