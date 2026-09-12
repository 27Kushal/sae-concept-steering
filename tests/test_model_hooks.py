"""Unit tests for configuration, device resolution, and steering hooks."""

import unittest
import torch

from src.config import ProjectConfig, get_device
from src.steering import ActivationSteeringHook


class TestModelHooks(unittest.TestCase):
    def test_config_profiles(self):
        """Verify small and full presets have expected scales."""
        cfg_small = ProjectConfig.create("small")
        cfg_full = ProjectConfig.create("full")

        self.assertEqual(cfg_small.scale, "small")
        self.assertEqual(cfg_full.scale, "full")
        self.assertLess(cfg_small.data.total_tokens, cfg_full.data.total_tokens)
        self.assertLess(cfg_small.train.total_steps, cfg_full.train.total_steps)

    def test_device_detection(self):
        """Verify device detection produces valid torch.device without throwing."""
        dev = get_device()
        self.assertIsInstance(dev, torch.device)
        self.assertIn(dev.type, ["cuda", "mps", "cpu"])

    def test_steering_hook_passthrough_when_alpha_zero(self):
        """Verify that alpha=0.0 hook returns exact input activation without change."""
        dummy_resid = torch.randn(2, 8, 768)
        steering_vec = torch.randn(768)

        hook = ActivationSteeringHook(steering_vector=steering_vec, alpha=0.0)
        output = hook(dummy_resid.clone(), None)

        self.assertTrue(torch.equal(output, dummy_resid))

    def test_steering_hook_vector_addition(self):
        """Verify that alpha != 0 adds scaled steering vector across tokens."""
        dummy_resid = torch.zeros(2, 4, 768)
        steering_vec = torch.ones(768)

        hook = ActivationSteeringHook(steering_vector=steering_vec, alpha=2.5)
        output = hook(dummy_resid, None)

        expected = torch.full((2, 4, 768), 2.5)
        self.assertTrue(torch.allclose(output, expected))


if __name__ == "__main__":
    unittest.main()
