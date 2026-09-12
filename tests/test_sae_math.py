"""Unit tests verifying SAE mathematical properties, norm constraints, and metrics."""

import os
import shutil
import tempfile
import unittest
import torch

from src.config import SAEArchitectureConfig
from src.sae import SparseAutoencoder


class TestSAEMath(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.config = SAEArchitectureConfig(
            d_model=64,
            expansion_factor=4,
            d_sae=256,
            l1_coefficient=1e-3,
        )
        self.sae = SparseAutoencoder(self.config)

    def test_unit_norm_constraint(self):
        """Verify decoder columns are strictly normalized to unit L2 norm."""
        # Corrupt decoder weights with arbitrary non-unit norms
        self.sae.W_dec.weight.data.mul_(5.5)

        # Enforce unit norm constraint
        self.sae.set_decoder_norm_to_unit()

        # Check norms across all columns (d_sae features)
        col_norms = torch.norm(self.sae.W_dec.weight.data, p=2, dim=0)
        expected_norms = torch.ones_like(col_norms)
        self.assertTrue(
            torch.allclose(col_norms, expected_norms, atol=1e-5),
            f"Decoder columns deviate from unit norm: max diff {torch.max(torch.abs(col_norms - 1.0))}",
        )

    def test_fraction_of_variance_explained(self):
        """Verify FVE yields 1.0 on exact reconstruction and <= 0.0 on noisy/constant reconstruction."""
        batch = torch.randn(100, 64)

        # Perfect reconstruction
        fve_perfect = SparseAutoencoder.compute_fve(batch, batch)
        self.assertAlmostEqual(fve_perfect, 1.0, places=5)

        # Mean predictor reconstruction
        mean_x = batch.mean(dim=0, keepdim=True).expand_as(batch)
        fve_mean = SparseAutoencoder.compute_fve(batch, mean_x)
        self.assertAlmostEqual(fve_mean, 0.0, places=3)

    def test_l0_metric(self):
        """Verify L0 metric correctly measures average number of active latents per token."""
        # Create synthetic activations: token 0 has 3 active, token 1 has 5 active
        acts = torch.zeros(2, 10)
        acts[0, [1, 3, 5]] = 2.0
        acts[1, [0, 2, 4, 6, 8]] = 1.5

        l0 = SparseAutoencoder.compute_l0(acts)
        self.assertEqual(l0, (3 + 5) / 2.0)

    def test_forward_and_backward(self):
        """Verify forward pass computes positive losses and gradients propagate."""
        batch = torch.randn(16, 64)
        out = self.sae(batch)

        self.assertGreater(out.l2_loss.item(), 0.0)
        self.assertGreater(out.l1_loss.item(), 0.0)
        self.assertGreater(out.loss.item(), 0.0)
        self.assertEqual(out.sae_out.shape, batch.shape)
        self.assertEqual(out.feature_acts.shape, (16, 256))

        # Backward pass
        out.loss.backward()
        self.assertIsNotNone(self.sae.W_enc.weight.grad)
        self.assertIsNotNone(self.sae.W_dec.weight.grad)
        self.assertIsNotNone(self.sae.b_dec.grad)

    def test_checkpoint_roundtrip(self):
        """Verify save_pretrained and from_pretrained preserve weights and config."""
        temp_dir = tempfile.mkdtemp()
        try:
            self.sae.save_pretrained(temp_dir)
            loaded_sae = SparseAutoencoder.from_pretrained(temp_dir)

            self.assertEqual(self.sae.d_model, loaded_sae.d_model)
            self.assertEqual(self.sae.d_sae, loaded_sae.d_sae)
            self.assertTrue(
                torch.allclose(self.sae.W_dec.weight, loaded_sae.W_dec.weight)
            )
            self.assertTrue(
                torch.allclose(self.sae.W_enc.weight, loaded_sae.W_enc.weight)
            )
        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    unittest.main()
