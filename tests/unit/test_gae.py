"""Tests for gae.py."""

import torch

from vrollout import gae


def test_pytorch_gae_shape_and_numerical_sanity():
    batch_size = 4
    seq_len = 8
    rewards = torch.zeros((batch_size, seq_len), dtype=torch.float32)
    rewards[:, -1] = 1.0  # terminal reward
    values = torch.ones((batch_size, seq_len), dtype=torch.float32) * 0.5
    next_values = torch.zeros((batch_size,), dtype=torch.float32)
    dones = torch.zeros((batch_size, seq_len), dtype=torch.float32)
    dones[:, -1] = 1.0

    advantages, target_values = gae.compute_gae(
        rewards, values, next_values, dones, gamma=0.99, lam=0.95
    )

    assert advantages.shape == (batch_size, seq_len)
    assert target_values.shape == (batch_size, seq_len)
    assert torch.allclose(target_values, advantages + values, atol=1e-5)
