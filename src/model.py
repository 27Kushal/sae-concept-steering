"""Model loading and TransformerLens wrapper with robust MPS/CPU/CUDA fallback."""

from __future__ import annotations
import logging
from typing import Optional, Tuple
import torch

try:
    from transformer_lens import HookedTransformer
except ImportError:
    HookedTransformer = None

from src.config import ModelConfig, get_device

logger = logging.getLogger(__name__)


def load_hooked_transformer(
    config: Optional[ModelConfig] = None,
    device: Optional[torch.device] = None,
) -> Tuple[HookedTransformer, torch.device]:
    """Loads a HookedTransformer model with robust device fallback.
    
    If TransformerLens is not installed or fails on MPS, falls back to CPU
    to guarantee flawless local execution without cryptic Metal errors.
    """
    if HookedTransformer is None:
        raise ImportError(
            "transformer_lens is not installed. Please install it via "
            "`pip install transformer_lens`."
        )

    if config is None:
        config = ModelConfig()

    resolved_device = device or config.resolved_device()

    # Attempt loading on resolved device
    try:
        logger.info(f"Loading {config.model_name} on {resolved_device}...")
        model = HookedTransformer.from_pretrained(
            config.model_name,
            device=resolved_device,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
        )
        model.eval()
        return model, resolved_device
    except Exception as e:
        if resolved_device.type == "mps":
            logger.warning(
                f"Loading on MPS failed with error: {e}. "
                "Falling back safely to CPU..."
            )
            model = HookedTransformer.from_pretrained(
                config.model_name,
                device="cpu",
                fold_ln=False,
                center_writing_weights=False,
                center_unembed=False,
            )
            model.eval()
            return model, torch.device("cpu")
        raise e
