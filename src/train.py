"""SAE training loop with metrics logging, unit-norm projection, and checkpointing."""

from __future__ import annotations
import os
import glob
import json
import logging
from typing import Dict, List, Optional
import torch
import torch.optim as optim
from tqdm import tqdm

from src.config import TrainingConfig, SAEArchitectureConfig, get_device
from src.sae import SparseAutoencoder
from src.data import ShardedActivationBuffer

logger = logging.getLogger(__name__)


def train_sae(
    sae: SparseAutoencoder,
    train_buffer: ShardedActivationBuffer,
    val_shard_path: Optional[str] = None,
    train_config: Optional[TrainingConfig] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, any]:
    """Executes the SAE training loop with unit-norm projection and metric tracking."""
    if train_config is None:
        train_config = TrainingConfig()

    resolved_device = device or get_device()
    sae = sae.to(resolved_device)
    sae.train()

    optimizer = optim.Adam(
        sae.parameters(),
        lr=train_config.lr,
        betas=train_config.betas,
        eps=train_config.eps,
    )

    # Learning rate linear warmup
    def get_lr(step: int) -> float:
        if step < train_config.warmup_steps:
            return train_config.lr * float(step + 1) / float(max(1, train_config.warmup_steps))
        return train_config.lr

    # Tracking metrics
    history: List[Dict[str, float]] = []
    feature_activation_counts = torch.zeros(sae.d_sae, device=resolved_device)
    total_eval_tokens = 0

    os.makedirs(train_config.checkpoint_dir, exist_ok=True)
    pbar = tqdm(total=train_config.total_steps, desc="Training SAE")

    step = 0
    buffer_iter = iter(train_buffer)

    while step < train_config.total_steps:
        try:
            batch = next(buffer_iter)
        except StopIteration:
            buffer_iter = iter(train_buffer)
            batch = next(buffer_iter)

        batch = batch.to(resolved_device)

        # Update learning rate
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad()
        output = sae(batch)
        output.loss.backward()

        # Remove parallel gradient component if desired
        sae.remove_gradient_parallel_to_decoder_directions()

        optimizer.step()

        # CRITICAL: Constrain decoder columns to unit L2 norm after EVERY optimizer step
        sae.set_decoder_norm_to_unit()

        # Track latent activations for dead feature monitoring
        with torch.no_grad():
            active_mask = (output.feature_acts > train_config.dead_feature_threshold)
            feature_activation_counts += active_mask.sum(dim=0)
            total_eval_tokens += batch.shape[0]

        step += 1
        pbar.update(1)

        # Periodic logging and evaluation
        if step % train_config.eval_every == 0 or step == train_config.total_steps:
            with torch.no_grad():
                l0 = SparseAutoencoder.compute_l0(output.feature_acts)
                fve = SparseAutoencoder.compute_fve(batch, output.sae_out)
                dead_features = (feature_activation_counts == 0).sum().item()
                dead_rate = dead_features / sae.d_sae

                metrics = {
                    "step": step,
                    "lr": lr,
                    "total_loss": output.loss.item(),
                    "l2_loss": output.l2_loss.item(),
                    "l1_loss": output.l1_loss.item(),
                    "l0": l0,
                    "fve": fve,
                    "dead_features": dead_features,
                    "dead_rate": dead_rate,
                }
                history.append(metrics)
                pbar.set_postfix(
                    l2=f"{output.l2_loss.item():.4f}",
                    l0=f"{l0:.1f}",
                    fve=f"{fve:.3f}",
                    dead=f"{dead_rate * 100:.1f}%",
                )

        # Periodic checkpointing
        if step % train_config.save_every == 0 or step == train_config.total_steps:
            ckpt_path = os.path.join(train_config.checkpoint_dir, f"step_{step}")
            sae.save_pretrained(ckpt_path)
            logger.info(f"Saved checkpoint to {ckpt_path}")

    pbar.close()

    # Final evaluation on held-out validation shard if available
    final_val_fve = None
    if val_shard_path and os.path.exists(val_shard_path):
        sae.eval()
        with torch.no_grad():
            val_batch = torch.load(val_shard_path, map_location=resolved_device)
            val_out = sae(val_batch)
            final_val_fve = SparseAutoencoder.compute_fve(val_batch, val_out.sae_out)
            final_val_l0 = SparseAutoencoder.compute_l0(val_out.feature_acts)
            logger.info(f"Held-out Validation FVE: {final_val_fve:.4f}, L0: {final_val_l0:.2f}")

    # Save final model and training summary
    final_dir = os.path.join(train_config.checkpoint_dir, "final")
    sae.save_pretrained(final_dir)

    summary = {
        "final_step": step,
        "final_loss": history[-1]["total_loss"] if history else None,
        "final_l2_loss": history[-1]["l2_loss"] if history else None,
        "final_l0": history[-1]["l0"] if history else None,
        "final_fve": history[-1]["fve"] if history else None,
        "validation_fve": final_val_fve,
        "dead_features": history[-1]["dead_features"] if history else None,
        "dead_feature_rate": history[-1]["dead_rate"] if history else None,
        "history": history,
    }

    with open(os.path.join(train_config.checkpoint_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        f"Training complete. Final L0: {summary['final_l0']:.2f}, "
        f"FVE: {summary['final_fve']:.4f}, "
        f"Dead Feature Rate: {summary['dead_feature_rate'] * 100:.2f}%"
    )
    return summary
