#!/usr/bin/env python3
"""CLI script for training Sparse Autoencoder on cached activations."""

import argparse
import glob
import logging
import os
import sys

# Ensure repository root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import ProjectConfig, get_device
from src.sae import SparseAutoencoder
from src.data import ShardedActivationBuffer
from src.train import train_sae

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train Sparse Autoencoder on cached activations.")
    parser.add_argument(
        "--scale",
        type=str,
        default="small",
        choices=["small", "full"],
        help="Configuration scale preset: 'small' (local prototype) or 'full' (Colab scale)",
    )
    parser.add_argument("--steps", type=int, default=None, help="Override total training steps")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--l1", type=float, default=None, help="L1 sparsity penalty coefficient")
    parser.add_argument("--shards_dir", type=str, default=None, help="Directory containing .pt shards")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to save checkpoints")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('mps', 'cpu', 'cuda')")
    args = parser.parse_args()

    config = ProjectConfig.create(scale=args.scale)
    if args.steps is not None:
        config.train.total_steps = args.steps
    if args.batch_size is not None:
        config.train.batch_size = args.batch_size
    if args.lr is not None:
        config.train.lr = args.lr
    if args.l1 is not None:
        config.sae.l1_coefficient = args.l1
    if args.checkpoint_dir is not None:
        config.train.checkpoint_dir = args.checkpoint_dir

    shards_dir = args.shards_dir or config.data.cache_dir
    shard_paths = sorted(glob.glob(os.path.join(shards_dir, "shard_*.pt")))
    if not shard_paths:
        raise FileNotFoundError(
            f"No activation shards found in {shards_dir}. "
            "Please run `python scripts/01_collect_activations.py` first."
        )

    device = get_device(args.device)
    logger.info(f"Initializing SAE training on {device} with {len(shard_paths)} shards...")

    sae = SparseAutoencoder(config.sae)
    train_buffer = ShardedActivationBuffer(
        shard_paths=shard_paths,
        batch_size=config.train.batch_size,
        device=device,
        shuffle=True,
    )

    summary = train_sae(
        sae=sae,
        train_buffer=train_buffer,
        val_shard_path=shard_paths[-1] if len(shard_paths) > 1 else None,
        train_config=config.train,
        device=device,
    )

    print("\n" + "=" * 60)
    print("SAE Training Finished Successfully!")
    print(f"Final L0:                {summary['final_l0']:.2f}")
    print(f"Fraction Variance Expl.: {summary['final_fve']:.4f}")
    print(f"Dead Feature Rate:       {summary['dead_feature_rate'] * 100:.2f}%")
    print(f"Checkpoint saved to:     {config.train.checkpoint_dir}/final")
    print("=" * 60)


if __name__ == "__main__":
    main()
