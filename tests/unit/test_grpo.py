import torch

from vrollout import grpo

torch.manual_seed(42)


def test_compute_group_advantages():
    rewards = torch.tensor([1.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    group_size = 3
    eps = 1e-6

    advantages = grpo.compute_group_advantages(rewards, group_size, eps)

    exp_advantages = torch.tensor([0.7071, -1.4142, 0.7071, 0.0, 0.0, 0.0])
    torch.testing.assert_close(advantages, exp_advantages, rtol=1e-5, atol=1e-5)


def test_compute_grpo_loss():
    batch_size = 2
    seq_len = 3
    log_probs = torch.full((batch_size, seq_len), -1.0)
    old_log_probs = torch.full((batch_size, seq_len), -1.0)
    ref_log_probs = torch.full((batch_size, seq_len), -1.0)
    advantages = torch.tensor([1.5, -1.0])
    mask = torch.ones((batch_size, seq_len), dtype=torch.bool)
    clip_eps = 0.2
    beta = 0.04

    loss, metrics = grpo.compute_grpo_loss(
        log_probs,
        old_log_probs,
        ref_log_probs,
        advantages,
        mask,
        clip_eps,
        beta,
    )

    exp_loss = torch.tensor(-advantages.mean())
    torch.testing.assert_close(loss, exp_loss, rtol=1e-5, atol=1e-5)
    assert metrics.loss == exp_loss.item()
    assert metrics.mean_kl == 0.0
    assert metrics.clip_fraction == 0.0
