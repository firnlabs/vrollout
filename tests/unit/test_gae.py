"""Tests for gae.py."""

import pytest
import torch

from vrollout import gae


def test_compute_gae_no_padding():
    batch_size = 4
    seq_len = 8
    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int64)
    rewards = torch.zeros((batch_size, seq_len), dtype=torch.float32)
    rewards[:, -1] = 1.0  # terminal reward
    values = torch.ones((batch_size, seq_len), dtype=torch.float32) * 0.5
    next_values = torch.zeros((batch_size,), dtype=torch.float32)
    dones = torch.zeros((batch_size, seq_len), dtype=torch.float32)
    dones[:, -1] = 1.0

    advantages, target_values = gae.compute_gae(
        rewards, values, next_values, dones, seq_lens, gamma=0.99, lam=0.95
    )

    assert advantages.shape == (batch_size, seq_len)
    assert target_values.shape == (batch_size, seq_len)
    assert torch.allclose(target_values, advantages + values, atol=1e-5)


def test_compute_gae_error_max_seq_len_zero():
    rewards = torch.tensor([])
    values = torch.zeros((3,))
    next_values = torch.zeros((1,))
    dones = torch.zeros_like(values)
    seq_lens = torch.tensor([])

    with pytest.raises(ValueError, match="Maximum sequence length is zero."):
        gae.compute_gae(rewards, values, next_values, dones, seq_lens)


def test_compute_gae_error_seq_lens_exceeds_max():
    batch_size = 3
    seq_len = 3
    rewards = torch.zeros((batch_size, seq_len))
    values = torch.zeros_like(rewards)
    next_values = torch.zeros((batch_size,))
    dones = torch.zeros_like(values)
    seq_lens = torch.tensor([3, 5, 4])

    with pytest.raises(ValueError, match="Maximum value"):
        gae.compute_gae(rewards, values, next_values, dones, seq_lens)


def test_compute_gae_with_variable_padding():
    gamma = 0.9
    lam = 0.8
    seq_lens = torch.tensor([2, 4, 1], dtype=torch.int64)

    rewards = torch.tensor(
        [
            [1.0, 2.0, 0.0, 0.0],
            [0.5, 0.5, 0.5, 1.0],
            [3.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    values = torch.tensor(
        [
            [0.5, 1.0, 99.0, 99.0],  # 99.0 in padding to verify no leakage
            [0.2, 0.3, 0.4, 0.5],
            [1.0, 99.0, 99.0, 99.0],
        ],
        dtype=torch.float32,
    )

    next_values = torch.tensor([2.0, 5.0, 7.0], dtype=torch.float32)

    dones = torch.tensor(
        [
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )

    advantages, target_values = gae.compute_gae(
        rewards, values, next_values, dones, seq_lens, gamma=gamma, lam=lam
    )

    exp_advantages = torch.tensor(
        [
            [3.416, 2.8, 0.0, 0.0],
            [1.444944, 1.2152, 0.91, 0.5],
            [2.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(advantages, exp_advantages, atol=1e-5, rtol=1e-5)

    # 3. Check padding positions are strictly zero for advantages and target_values
    assert advantages[0, 2:].tolist() == [0.0, 0.0]
    assert target_values[0, 2:].tolist() == [0.0, 0.0]
    assert advantages[2, 1:].tolist() == [0.0, 0.0, 0.0]
    assert target_values[2, 1:].tolist() == [0.0, 0.0, 0.0]

    # 4. Check targets == advantages + values for valid tokens
    assert torch.allclose(
        target_values[0, :2], advantages[0, :2] + values[0, :2], atol=1e-5
    )
    assert torch.allclose(target_values[1], advantages[1] + values[1], atol=1e-5)
    assert torch.allclose(
        target_values[2, :1], advantages[2, :1] + values[2, :1], atol=1e-5
    )


@pytest.mark.skipif(not gae.TRITON_AVAILABLE, reason="Requires OpenAI Triton")
class TestTritonGAE:
    def test_numerical_parity_with_pytorch_age(self):
        torch.manual_seed(42)
        B, T = 8, 64
        rewards = torch.randn((B, T), device="cuda", dtype=torch.float32)
        values = torch.randn((B, T), device="cuda", dtype=torch.float32)
        next_values = torch.randn((B,), device="cuda", dtype=torch.float32)
        dones = (torch.rand((B, T), device="cuda") > 0.9).float()

        adv_torch, ret_torch = gae.compute_gae_torch(
            rewards, values, next_values, dones
        )
        adv_triton, ret_triton = gae.compute_gae_triton(
            rewards, values, next_values, dones
        )

        torch.testing.allclose(adv_triton, adv_torch, atol=1e-5, rtol=1e-5)
        torch.testing.allclose(ret_triton, ret_torch, atol=1e-5, rtol=1e-5)
