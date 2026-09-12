"""Activation collection, sharding, and memory-efficient dataset buffers."""

from __future__ import annotations
import os
import glob
import logging
from typing import Generator, List, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.config import DataCollectionConfig, ModelConfig
from src.model import load_hooked_transformer

logger = logging.getLogger(__name__)


def is_network_available(timeout: float = 0.5) -> bool:
    import socket
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("8.8.8.8", 53))
        return True
    except (socket.timeout, OSError):
        return False


def get_text_dataset(config: DataCollectionConfig) -> Generator[str, None, None]:
    """Streams text documents from Hugging Face datasets with local fallback."""
    if is_network_available():
        try:
            from datasets import load_dataset
            ds = load_dataset(
                config.dataset_name,
                config.dataset_config,
                split=config.split,
                streaming=True,
            )
            for row in ds:
                text = row.get("text", "")
                if len(text.strip()) > 50:
                    yield text
            return
        except Exception as e:
            logger.warning(f"Could not stream {config.dataset_name}: {e}. Using fallback corpus.")
    else:
        logger.info("Offline mode detected: using built-in reference corpus.")
        # Fallback text generator for offline/quick testing
        sample_corpus = [
            "Sparse autoencoders decompose language model activations into interpretable features.",
            "Mechanistic interpretability aims to reverse engineer the computational graph of neural networks.",
            "Python is a versatile programming language widely used in data science and machine learning.",
            "Activation steering allows steering generation by intervening on the residual stream.",
            "Linear representations in neural networks exhibit superposition due to limited feature capacity.",
            "Transformers utilize self-attention mechanisms to capture long-range contextual dependencies.",
            "Monosemanticity refers to neurons or features having single, well-defined conceptual meanings.",
            "Residual streams act as a shared communication channel across all transformer attention blocks.",
        ] * 200
        for text in sample_corpus:
            yield text


def collect_and_shard_activations(
    data_config: DataCollectionConfig,
    model_config: ModelConfig,
    device: Optional[torch.device] = None,
) -> List[str]:
    """Extracts residual stream activations and writes them to disk in shards.
    
    Returns a list of saved shard file paths.
    """
    os.makedirs(data_config.cache_dir, exist_ok=True)
    model, resolved_device = load_hooked_transformer(model_config, device)
    tokenizer = model.tokenizer

    target_tokens = data_config.total_tokens
    shard_size = data_config.shard_size_tokens
    context_len = data_config.context_length
    batch_size = data_config.batch_size
    hook_name = model_config.hook_name

    logger.info(
        f"Starting activation collection: target={target_tokens} tokens, "
        f"shard_size={shard_size}, hook={hook_name} on {resolved_device}"
    )

    text_gen = get_text_dataset(data_config)
    tokens_collected = 0
    shard_idx = 0
    shard_paths: List[str] = []

    current_shard_acts: List[torch.Tensor] = []
    current_shard_tokens = 0

    batch_texts: List[str] = []
    pbar = tqdm(total=target_tokens, desc="Collecting activations")

    while tokens_collected < target_tokens:
        # Accumulate text batch
        while len(batch_texts) < batch_size:
            try:
                text = next(text_gen)
                batch_texts.append(text)
            except StopIteration:
                text_gen = get_text_dataset(data_config)
                text = next(text_gen)
                batch_texts.append(text)

        # Tokenize with truncation / padding to fixed context length
        encodings = tokenizer(
            batch_texts,
            max_length=context_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encodings["input_ids"].to(resolved_device)
        attention_mask = encodings["attention_mask"].to(resolved_device)
        batch_texts.clear()

        # Run model with cache
        with torch.no_grad():
            _, cache = model.run_with_cache(
                input_ids,
                names_filter=lambda name: name == hook_name,
            )
            # Activations shape: [batch_size, context_len, d_model]
            acts = cache[hook_name]

            # Filter only valid non-padding tokens
            mask_flat = attention_mask.bool().flatten()
            acts_flat = acts.reshape(-1, acts.shape[-1])[mask_flat]

            # Move to CPU for disk caching
            acts_cpu = acts_flat.detach().cpu().to(torch.float32)

        n_new_tokens = acts_cpu.shape[0]
        if n_new_tokens == 0:
            continue

        # Trim if we exceed target tokens
        if tokens_collected + n_new_tokens > target_tokens:
            needed = target_tokens - tokens_collected
            acts_cpu = acts_cpu[:needed]
            n_new_tokens = needed

        current_shard_acts.append(acts_cpu)
        current_shard_tokens += n_new_tokens
        tokens_collected += n_new_tokens
        pbar.update(n_new_tokens)

        # Save shard if threshold reached or final target reached
        if current_shard_tokens >= shard_size or tokens_collected >= target_tokens:
            shard_tensor = torch.cat(current_shard_acts, dim=0)
            shard_path = os.path.join(
                data_config.cache_dir, f"shard_{shard_idx:05d}.pt"
            )
            torch.save(shard_tensor, shard_path)
            shard_paths.append(shard_path)
            logger.info(
                f"Saved shard {shard_idx}: {shard_tensor.shape[0]} tokens -> {shard_path}"
            )
            shard_idx += 1
            current_shard_acts.clear()
            current_shard_tokens = 0

    pbar.close()
    logger.info(
        f"Activation collection complete: {tokens_collected} tokens saved across {len(shard_paths)} shards."
    )
    return shard_paths


class ShardedActivationBuffer:
    """Iterates through disk-sharded activation files, shuffling and yielding batches.
    
    Avoids loading the entire multi-gigabyte activation dataset into RAM at once.
    """

    def __init__(
        self,
        shard_paths: List[str],
        batch_size: int = 2048,
        device: Optional[torch.device] = None,
        shuffle: bool = True,
    ):
        self.shard_paths = sorted(shard_paths)
        if not self.shard_paths:
            raise ValueError("No shard paths provided to ShardedActivationBuffer.")
        self.batch_size = batch_size
        self.device = device or torch.device("cpu")
        self.shuffle = shuffle

    def __iter__(self) -> Generator[torch.Tensor, None, None]:
        shard_order = list(self.shard_paths)
        if self.shuffle:
            import random
            random.shuffle(shard_order)

        for path in shard_order:
            shard = torch.load(path, map_location="cpu")
            if self.shuffle:
                perm = torch.randperm(shard.shape[0])
                shard = shard[perm]

            num_batches = shard.shape[0] // self.batch_size
            for i in range(num_batches):
                batch = shard[i * self.batch_size : (i + 1) * self.batch_size]
                yield batch.to(self.device)
