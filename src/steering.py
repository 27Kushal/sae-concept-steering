"""Activation steering implementation: SAE feature injection and Difference-of-Means baseline."""

from __future__ import annotations
import logging
from typing import List, Optional, Tuple, Dict, Any
import torch
import torch.nn.functional as F

from src.sae import SparseAutoencoder

logger = logging.getLogger(__name__)


CONCEPTS = {
    "python_code": {
        "feature_idx": 42,  # Target feature index (or determined from Part 3)
        "test_prompts": [
            "Write a function that",
            "Here is the implementation:",
            "To solve this problem in software,",
        ],
        "positive_prompts": [
            "def calculate_mean(numbers):\n    return sum(numbers) / len(numbers)",
            "import os\nimport sys\nfrom pathlib import Path",
            "class NeuralNet(torch.nn.Module):\n    def __init__(self):",
            "for i in range(len(items)):\n    if items[i] == target: return i",
        ],
        "negative_prompts": [
            "The morning sun cast a gentle golden glow over the tranquil valley.",
            "Historical trade routes connected civilizations across ancient continents.",
            "Fresh ingredients and careful seasoning are the secrets to great cuisine.",
            "The orchestra played a moving symphony that echoed through the grand concert hall.",
        ],
    },
    "sentiment": {
        "feature_idx": 15,
        "test_prompts": [
            "The movie was",
            "The customer service was",
            "My experience at the restaurant was",
        ],
        "positive_prompts": [
            "This is absolutely wonderful and brilliant, an outstanding masterpiece!",
            "I love this incredible and fantastic experience, truly superb and joyful!",
            "Delightful, excellent service and amazingly kind staff.",
        ],
        "negative_prompts": [
            "This is terrible, horrible, completely disappointing and dreadful.",
            "I hate this awful and ugly experience, truly the worst service.",
            "Disgusting, poor quality and horribly rude staff.",
        ],
    },
}


class ActivationSteeringHook:
    """TransformerLens hook for adding a steering vector to residual stream activations.
    
    Formula:
        h_steered = h_original + alpha * steering_vector
    """

    def __init__(
        self,
        steering_vector: torch.Tensor,
        alpha: float = 0.0,
        apply_to_last_token_only: bool = False,
    ):
        self.steering_vector = steering_vector
        self.alpha = alpha
        self.apply_to_last_token_only = apply_to_last_token_only

    def __call__(self, value: torch.Tensor, hook: Any) -> torch.Tensor:
        """Hook function called by TransformerLens.
        
        Args:
            value: Tensor of shape [batch_size, seq_len, d_model]
            hook: TransformerLens hook context
        """
        if self.alpha == 0.0 or self.steering_vector is None:
            return value

        vec = self.steering_vector.to(value.device, dtype=value.dtype)
        # Reshape to broadcast across batch and seq_len: [1, 1, d_model]
        vec = vec.view(1, 1, -1)

        if self.apply_to_last_token_only:
            # Modify only current predicting token
            value[:, -1:, :] = value[:, -1:, :] + self.alpha * vec
            return value

        return value + self.alpha * vec


def compute_difference_of_means_vector(
    model: Any,
    positive_prompts: List[str],
    negative_prompts: List[str],
    hook_name: str = "blocks.6.hook_resid_post",
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Computes a normalized Difference-of-Means activation steering vector.
    
    Formula:
        v_diff = mean(h_pos) - mean(h_neg)
        v_hat = v_diff / ||v_diff||_2
    """
    resolved_device = device or next(model.parameters()).device

    def get_prompt_activations(prompts: List[str]) -> List[torch.Tensor]:
        acts = []
        for prompt in prompts:
            tokens = model.tokenizer.encode(prompt, return_tensors="pt").to(resolved_device)
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, names_filter=lambda n: n == hook_name)
                # Take last token residual stream representation
                resid_last = cache[hook_name][0, -1, :].detach().cpu()
                acts.append(resid_last)
        return acts

    pos_acts = torch.stack(get_prompt_activations(positive_prompts))
    neg_acts = torch.stack(get_prompt_activations(negative_prompts))

    mean_pos = pos_acts.mean(dim=0)
    mean_neg = neg_acts.mean(dim=0)

    diff_vector = mean_pos - mean_neg
    norm = torch.norm(diff_vector, p=2)
    normalized_diff_vector = diff_vector / max(norm.item(), 1e-8)

    logger.info(f"Computed Difference-of-Means vector: raw norm was {norm.item():.4f}")
    return normalized_diff_vector


def generate_with_steering(
    model: Any,
    prompt: str,
    steering_vector: Optional[torch.Tensor],
    alpha: float,
    hook_name: str = "blocks.6.hook_resid_post",
    max_new_tokens: int = 35,
    temperature: float = 0.7,
    top_p: float = 0.9,
    device: Optional[torch.device] = None,
) -> str:
    """Generates text from base LLM while applying activation steering hook at target layer."""
    resolved_device = device or next(model.parameters()).device

    hook = ActivationSteeringHook(steering_vector=steering_vector, alpha=alpha)

    # Convert prompt to tokens
    input_ids = model.tokenizer.encode(prompt, return_tensors="pt").to(resolved_device)

    with torch.no_grad():
        if alpha == 0.0 or steering_vector is None:
            # Unsteered baseline generation
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                stop_at_eos=False,
                verbose=False,
            )
        else:
            # Steered generation with hooked residual stream
            with model.hooks(fwd_hooks=[(hook_name, hook)]):
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop_at_eos=False,
                    verbose=False,
                )

    return model.tokenizer.decode(output_ids[0], skip_special_tokens=True)


def run_steering_sweep(
    model: Any,
    prompts: List[str],
    steering_vector: torch.Tensor,
    alphas: List[float],
    hook_name: str = "blocks.6.hook_resid_post",
    max_new_tokens: int = 35,
    num_samples_per_prompt: int = 2,
    device: Optional[torch.device] = None,
) -> List[Dict[str, Any]]:
    """Runs a sweep over steering coefficients alpha across a set of prompts."""
    results: List[Dict[str, Any]] = []

    for prompt in prompts:
        for alpha in alphas:
            samples = []
            for _ in range(num_samples_per_prompt):
                generated_text = generate_with_steering(
                    model=model,
                    prompt=prompt,
                    steering_vector=steering_vector,
                    alpha=alpha,
                    hook_name=hook_name,
                    max_new_tokens=max_new_tokens,
                    device=device,
                )
                samples.append(generated_text)

            results.append({
                "prompt": prompt,
                "alpha": alpha,
                "samples": samples,
            })

    return results
