#!/usr/bin/env python3
"""CLI script for activation steering and rigorous quantitative evaluation vs Difference-of-Means."""

import argparse
import logging
import os
import sys

# Ensure repository root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import pandas as pd

from src.config import ProjectConfig, get_device
from src.model import load_hooked_transformer
from src.sae import SparseAutoencoder
from src.steering import (
    CONCEPTS,
    compute_difference_of_means_vector,
    run_steering_sweep,
)
from src.evaluate import run_full_comparative_evaluation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Activation steering and quantitative benchmark.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/small/final", help="Path to SAE checkpoint")
    parser.add_argument("--concept", type=str, default="python_code", choices=list(CONCEPTS.keys()))
    parser.add_argument("--feature_idx", type=int, default=None, help="Specific feature latent index to steer")
    parser.add_argument("--alphas", type=float, nargs="+", default=[-4.0, 0.0, 4.0, 8.0])
    parser.add_argument("--output_csv", type=str, default="results/evaluation_metrics.csv")
    parser.add_argument("--device", type=str, default=None, help="Device to use")
    args = parser.parse_args()

    device = get_device(args.device)
    concept_data = CONCEPTS[args.concept]
    feat_idx = args.feature_idx if args.feature_idx is not None else concept_data["feature_idx"]

    # 1. Load model and SAE
    logger.info(f"Loading HookedTransformer gpt2 on {device}...")
    model, _ = load_hooked_transformer(device=device)

    logger.info(f"Loading SAE from {args.checkpoint}...")
    sae = SparseAutoencoder.from_pretrained(args.checkpoint, device=device)

    # 2. Extract steering vectors
    # A) SAE Feature Direction: unit L2 norm decoder column
    sae_vector = sae.get_feature_direction(feat_idx).to(device)

    # B) Difference-of-Means Baseline Vector
    logger.info("Computing Difference-of-Means baseline vector...")
    diff_vector = compute_difference_of_means_vector(
        model=model,
        positive_prompts=concept_data["positive_prompts"],
        negative_prompts=concept_data["negative_prompts"],
        device=device,
    ).to(device)

    # 3. Generate steered samples
    logger.info(f"Generating SAE steered samples for concept '{args.concept}' (feature {feat_idx})...")
    sae_samples = run_steering_sweep(
        model=model,
        prompts=concept_data["test_prompts"],
        steering_vector=sae_vector,
        alphas=args.alphas,
        num_samples_per_prompt=2,
        device=device,
    )

    logger.info(f"Generating Difference-of-Means baseline samples for '{args.concept}'...")
    diff_samples = run_steering_sweep(
        model=model,
        prompts=concept_data["test_prompts"],
        steering_vector=diff_vector,
        alphas=args.alphas,
        num_samples_per_prompt=2,
        device=device,
    )

    # 4. Quantitative evaluation with independent judge and perplexity
    logger.info("Evaluating efficacy and collateral damage...")
    results_df = run_full_comparative_evaluation(
        model=model,
        concept_name=args.concept,
        sae_samples=sae_samples,
        diff_of_means_samples=diff_samples,
        device=device,
    )

    # Grouped summary table
    summary_df = results_df.groupby(["method", "alpha"]).agg({
        "judge_score": "mean",
        "perplexity": "mean",
        "perplexity_delta": "mean",
    }).reset_index()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    results_df.to_csv(args.output_csv, index=False)
    summary_csv = args.output_csv.replace(".csv", "_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    print("\n" + "=" * 75)
    print("Quantitative Steering Evaluation Summary (SAE vs Difference-of-Means):")
    print("=" * 75)
    print(summary_df.to_string(index=False))
    print("=" * 75)
    print(f"Detailed output saved to: {args.output_csv}")
    print(f"Summary saved to:         {summary_csv}")


if __name__ == "__main__":
    main()
