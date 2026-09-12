"""Systematic feature interpretation: top-k harvesting, hypothesis generation, and validation."""

from __future__ import annotations
import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
from tqdm import tqdm

from src.sae import SparseAutoencoder
from src.model import load_hooked_transformer

logger = logging.getLogger(__name__)


def sample_feature_subset(
    sae: SparseAutoencoder,
    val_acts: torch.Tensor,
    n_top: int = 50,
    n_random: int = 50,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """Selects top-activating and random features across validation activations.
    
    Returns:
        (top_feature_indices, random_feature_indices)
    """
    torch.manual_seed(seed)
    with torch.no_grad():
        # Encode validation activations
        feature_acts = sae.encode(val_acts)  # [N, d_sae]
        mean_acts = feature_acts.mean(dim=0)  # [d_sae]

        # Top-N most active features
        top_indices = torch.topk(mean_acts, k=min(n_top, sae.d_sae)).indices.tolist()

        # Random sample excluding already selected top features
        available_indices = list(set(range(sae.d_sae)) - set(top_indices))
        if len(available_indices) > 0:
            rand_perm = torch.randperm(len(available_indices))
            random_indices = [available_indices[i] for i in rand_perm[:n_random].tolist()]
        else:
            random_indices = []

    return top_indices, random_indices


def harvest_top_activating_snippets(
    model: Any,
    sae: SparseAutoencoder,
    texts: List[str],
    target_feature_indices: List[int],
    hook_name: str = "blocks.6.hook_resid_post",
    k: int = 5,
    context_window: int = 5,
    device: Optional[torch.device] = None,
) -> Dict[int, List[Dict[str, Any]]]:
    """Finds top-k text snippets that produce the highest activation for each target feature."""
    resolved_device = device or next(sae.parameters()).device
    sae.eval()

    # Data structure: {feature_idx: list of top k snippets}
    feature_snippets: Dict[int, List[Dict[str, Any]]] = {
        idx: [] for idx in target_feature_indices
    }
    target_set = set(target_feature_indices)

    tokenizer = model.tokenizer

    for text in tqdm(texts, desc="Harvesting feature activations"):
        tokens = tokenizer.encode(text, return_tensors="pt").to(resolved_device)
        if tokens.shape[1] < 4:
            continue

        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=lambda name: name == hook_name,
            )
            # shape: [1, seq_len, d_model]
            resid = cache[hook_name][0]  # [seq_len, d_model]
            feature_acts = sae.encode(resid)  # [seq_len, d_sae]

            token_strs = [model.to_string(t) for t in tokens[0]]

            for feat_idx in target_set:
                act_col = feature_acts[:, feat_idx]  # [seq_len]
                max_act, max_pos = act_col.max(dim=0)
                max_act_val = max_act.item()

                if max_act_val > 0.0:
                    pos = max_pos.item()
                    start = max(0, pos - context_window)
                    end = min(len(token_strs), pos + context_window + 1)
                    snippet = "".join(token_strs[start:end])
                    activating_token = token_strs[pos]

                    feature_snippets[feat_idx].append({
                        "activation": max_act_val,
                        "token": activating_token,
                        "token_pos": pos,
                        "snippet": snippet,
                    })

    # Sort each feature's snippets by activation descending, keep top k
    for feat_idx in feature_snippets:
        feature_snippets[feat_idx].sort(key=lambda x: x["activation"], reverse=True)
        feature_snippets[feat_idx] = feature_snippets[feat_idx][:k]

    return feature_snippets


def generate_feature_hypothesis(
    feature_idx: int,
    top_snippets: List[Dict[str, Any]],
    llm_api_fn: Optional[Any] = None,
) -> str:
    """Generates a concise concept hypothesis for what pattern the feature detects.
    
    Supports an external LLM API callback or falls back to robust local heuristic pattern analysis.
    """
    if not top_snippets:
        return "Inactive / Dead Feature: no activations observed across sample corpus."

    # If an external LLM caller is supplied, use it
    if llm_api_fn is not None:
        try:
            prompt = (
                f"A sparse autoencoder latent fires strongly on the following text snippets:\n"
                + "\n".join([f"- Token: '{s['token']}' in snippet: \"{s['snippet']}\" (act={s['activation']:.2f})" for s in top_snippets])
                + "\nState in 5-10 words what linguistic or semantic concept this feature represents."
            )
            return llm_api_fn(prompt).strip()
        except Exception as e:
            logger.warning(f"LLM API hypothesis generation failed: {e}. Using local heuristic.")

    # Local deterministic pattern analyzer
    tokens = [s["token"].strip() for s in top_snippets]
    lower_tokens = [t.lower() for t in tokens if t]

    # Heuristic pattern checks
    if all(t.isnumeric() for t in lower_tokens if t):
        return f"Numeric digits and numerical formatting (e.g., {', '.join(tokens[:3])})"
    if any(t in ["def", "import", "class", "return", "(", ")", ":", "=", "{", "}", ";", "print"] for t in lower_tokens):
        return f"Programming syntax and code delimiters (e.g., {', '.join(tokens[:3])})"
    if any(t in ["the", "a", "an", "this", "that", "these"] for t in lower_tokens):
        return f"Determiners and definite/indefinite articles (e.g., {', '.join(tokens[:3])})"
    if any(t in [".", ",", "!", "?", "\"", "'", "-"] for t in tokens):
        return f"Punctuation and clause boundary markers (e.g., {', '.join(tokens[:3])})"
    if all(t.isupper() for t in tokens if len(t) > 1):
        return f"Acronyms and capitalized abbreviations (e.g., {', '.join(tokens[:3])})"

    # Fallback description
    unique_tokens = list(dict.fromkeys(tokens))[:4]
    return f"Lexical association with tokens: {', '.join(repr(t) for t in unique_tokens)}"


def validate_feature_hypothesis(
    model: Any,
    sae: SparseAutoencoder,
    feature_idx: int,
    positive_prompts: List[str],
    negative_prompts: List[str],
    hook_name: str = "blocks.6.hook_resid_post",
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Validates whether a feature selectively fires on positive vs negative contrastive prompts.
    
    Computes mean positive activation, mean negative activation, and interpretability specificity score.
    """
    resolved_device = device or next(sae.parameters()).device
    sae.eval()

    def get_max_acts(prompts: List[str]) -> List[float]:
        acts = []
        for p in prompts:
            tokens = model.tokenizer.encode(p, return_tensors="pt").to(resolved_device)
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, names_filter=lambda n: n == hook_name)
                resid = cache[hook_name][0]
                feat_acts = sae.encode(resid)[:, feature_idx]
                acts.append(feat_acts.max().item())
        return acts

    pos_acts = get_max_acts(positive_prompts)
    neg_acts = get_max_acts(negative_prompts)

    mean_pos = float(np.mean(pos_acts)) if pos_acts else 0.0
    mean_neg = float(np.mean(neg_acts)) if neg_acts else 0.0

    # Specificity ratio: contrast between positive and negative activation
    specificity = (mean_pos - mean_neg) / max(mean_pos, 1e-4)

    return {
        "mean_positive_act": mean_pos,
        "mean_negative_act": mean_neg,
        "specificity": float(np.clip(specificity, -1.0, 1.0)),
        "pos_firing_rate": float(np.mean([a > 0 for a in pos_acts])),
        "neg_firing_rate": float(np.mean([a > 0 for a in neg_acts])),
    }


def run_systematic_feature_interpretation(
    model: Any,
    sae: SparseAutoencoder,
    val_acts: torch.Tensor,
    sample_texts: List[str],
    output_report_path: str = "results/feature_interpretability_report.json",
    n_top: int = 50,
    n_random: int = 50,
    hook_name: str = "blocks.6.hook_resid_post",
) -> List[Dict[str, Any]]:
    """Runs end-to-end systematic feature interpretation on 100 features.
    
    Saves and returns complete results including messy/uninterpretable features.
    """
    logger.info("Sampling 100 features (top 50 active + 50 random)...")
    top_indices, random_indices = sample_feature_subset(
        sae, val_acts, n_top=n_top, n_random=n_random
    )

    all_sampled = [(idx, "top_active") for idx in top_indices] + [
        (idx, "random") for idx in random_indices
    ]
    sampled_indices = [idx for idx, _ in all_sampled]

    logger.info("Harvesting top activating snippets...")
    feature_snippets = harvest_top_activating_snippets(
        model=model,
        sae=sae,
        texts=sample_texts,
        target_feature_indices=sampled_indices,
        hook_name=hook_name,
    )

    # Standard contrast validation prompts
    contrast_pos = [
        "def calculate_total(items):\n    return sum(items)",
        "import math\nx = math.sqrt(16)",
        "class DataProcessor:\n    def __init__(self):\n        pass",
    ]
    contrast_neg = [
        "The ancient forest was quiet under the pale moonlight.",
        "She decided to travel across the mountains during spring.",
        "A peaceful afternoon in the quiet village garden.",
    ]

    report: List[Dict[str, Any]] = []

    for feat_idx, sample_type in tqdm(all_sampled, desc="Interpreting features"):
        snippets = feature_snippets.get(feat_idx, [])
        hypothesis = generate_feature_hypothesis(feat_idx, snippets)

        # Quantitative contrast validation
        val_metrics = validate_feature_hypothesis(
            model=model,
            sae=sae,
            feature_idx=feat_idx,
            positive_prompts=contrast_pos,
            negative_prompts=contrast_neg,
            hook_name=hook_name,
        )

        max_val = snippets[0]["activation"] if snippets else 0.0
        tokens_seen = [s["token"] for s in snippets[:3]]

        # Honest classification: mark whether feature is cleanly interpretable, polysemantic, or dead
        if not snippets or max_val == 0.0:
            status = "dead"
        elif len(set(tokens_seen)) <= 2 and max_val > 1.0:
            status = "interpretable"
        else:
            status = "polysemantic_or_noisy"

        entry = {
            "feature_idx": feat_idx,
            "sample_type": sample_type,
            "status": status,
            "max_activation": round(max_val, 4),
            "hypothesis": hypothesis,
            "top_tokens": tokens_seen,
            "validation_metrics": val_metrics,
            "snippets": snippets[:3],
        }
        report.append(entry)

    os.makedirs(os.path.dirname(output_report_path), exist_ok=True)
    with open(output_report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(
        f"Saved full interpretability report across {len(report)} features to {output_report_path}"
    )
    return report
