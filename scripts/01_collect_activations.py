#!/usr/bin/env python3
"""CLI script for collecting and sharding residual stream activations."""

import argparse
import logging
import os
import sys

# Ensure repository root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import ProjectConfig, get_device
from src.data import collect_and_shard_activations

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Collect and cache residual stream activations.")
    parser.add_argument(
        "--scale",
        type=str,
        default="small",
        choices=["small", "full"],
        help="Configuration scale preset: 'small' (local prototype) or 'full' (Colab scale)",
    )
    parser.add_argument("--tokens", type=int, default=None, help="Override total tokens to collect")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('mps', 'cpu', 'cuda')")
    parser.add_argument("--cache_dir", type=str, default=None, help="Output directory for shards")
    args = parser.parse_args()

    config = ProjectConfig.create(scale=args.scale)
    if args.tokens is not None:
        config.data.total_tokens = args.tokens
    if args.cache_dir is not None:
        config.data.cache_dir = args.cache_dir

    device = get_device(args.device)
    logger.info(f"Running activation collection with scale='{args.scale}' on {device}")

    shards = collect_and_shard_activations(
        data_config=config.data,
        model_config=config.model,
        device=device,
    )

    print(f"\nSuccessfully collected {config.data.total_tokens} tokens into {len(shards)} shards.")
    for s in shards:
        print(f" - {s}")


if __name__ == "__main__":
    main()
