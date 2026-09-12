#!/usr/bin/env python3
"""CLI script for systematic (non-cherry-picked) feature interpretation."""

import argparse
import glob
import json
import logging
import os
import sys

# Ensure repository root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.config import ProjectConfig, get_device
from src.model import load_hooked_transformer
from src.sae import SparseAutoencoder
from src.interpret import run_systematic_feature_interpretation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Systematic feature interpretation across sampled latents.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/small/final", help="Path to SAE checkpoint")
    parser.add_argument("--shards_dir", type=str, default="data/activations_small", help="Path to activation shards")
    parser.add_argument("--output_report", type=str, default="results/feature_interpretability_report.json")
    parser.add_argument("--num_top", type=int, default=50, help="Number of top active features to sample")
    parser.add_argument("--num_random", type=int, default=50, help="Number of random features to sample")
    parser.add_argument("--device", type=str, default=None, help="Device to use")
    args = parser.parse_args()

    device = get_device(args.device)

    # 1. Load model and SAE
    logger.info(f"Loading HookedTransformer gpt2 on {device}...")
    model, _ = load_hooked_transformer(device=device)

    logger.info(f"Loading SAE from {args.checkpoint}...")
    sae = SparseAutoencoder.from_pretrained(args.checkpoint, device=device)

    # 2. Load validation activations
    shard_paths = sorted(glob.glob(os.path.join(args.shards_dir, "shard_*.pt")))
    if not shard_paths:
        raise FileNotFoundError(f"No shard files found in {args.shards_dir}")

    val_acts = torch.load(shard_paths[0], map_location=device)[:4096]

    # Sample texts for context extraction
    sample_texts = [
        "Python functions are defined using the def keyword, followed by the function name and parameters.",
        "A sparse autoencoder uses an overcomplete basis to represent linear directions in activation space.",
        "Deep neural networks exhibit superposition, representing more features than there are dimensions.",
        "The quick brown fox jumps over the lazy dog near the peaceful countryside river.",
        "class ModelTrainer:\n    def __init__(self, lr=0.001):\n        self.lr = lr",
        "Recent progress in mechanistic interpretability focuses on dictionary learning and steering vectors.",
        "import torch\nimport numpy as np\nimport matplotlib.pyplot as plt",
        "The historical monuments in Europe attract millions of tourists from around the world each year.",
    ] * 5

    # 3. Run systematic evaluation
    report = run_systematic_feature_interpretation(
        model=model,
        sae=sae,
        val_acts=val_acts,
        sample_texts=sample_texts,
        output_report_path=args.output_report,
        n_top=args.num_top,
        n_random=args.num_random,
    )

    # Summary breakdown
    statuses = [r["status"] for r in report]
    print("\n" + "=" * 60)
    print("Feature Interpretation Summary Across All Sampled Features:")
    print(f"Total Features Analyzed:  {len(report)}")
    print(f"Cleanly Interpretable:   {statuses.count('interpretable')}")
    print(f"Polysemantic / Noisy:    {statuses.count('polysemantic_or_noisy')}")
    print(f"Dead Features:           {statuses.count('dead')}")
    print(f"Full report written to:   {args.output_report}")
    print("=" * 60)


if __name__ == "__main__":
    main()
