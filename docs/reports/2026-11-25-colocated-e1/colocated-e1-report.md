# Single-GPU colocated E1

**Date:** November 25, 2026

## Summary

[E1](../2026-11-06-e0-e1-grpo-gsm8k/e1-report.md) trained synchronously on 2× A40: one GPU for the trainer and one for vLLM rollouts.
In synchronous mode each GPU sits idle while the other works, so the pair does about one GPU's worth of work.
To test this, E1's command ran for 30 steps with vLLM and the trainer sharing one A40.
The median step took 50.0 s, against 52.7 s for E1's first 30 steps on two A40s, with no out-of-memory errors or vLLM preemptions.
Synchronous training therefore needs only one GPU.
The second GPU pays off only with `--async-rollouts`, which overlaps rollout and training ([E2](../2026-11-18-e2-async-grpo-gsm8k/e2-report.md): 34.8 s).

## Background

E1 was the synchronous baseline for E2, which needs a separate rollout GPU to overlap rollout with training.
Running E1 on the same 2× A40 layout as E2 made `--async-rollouts` the only difference between the two runs, so their step times compare directly.
Whether the trainer and vLLM fit on one GPU without slowing down was also not known until this test.

## Setup

E1's training command ran on 1× A40 48 GB at commit `d4ae5eb`, with the same software as the original E1 run (PyTorch 2.13.0, vLLM 0.27.1, Ray 2.57.0).
Two flags changed:

- `--gpus-per-worker 0.5`: the vLLM worker requests half the GPU, as recommended when it shares the trainer's GPU (with only one GPU visible, vrollout places it there either way).
- `--gpu-memory-utilization 0.5`: vLLM's KV cache leaves room for the trainer.

Evals were turned off (`--eval-every 0`) since step time excludes them.
The comparison is with E1's first 30 steps rather than its 200-step median, because completion lengths change during training.

## Results

| | Median step | Rollout | Training | Weight sync |
| --- | --- | --- | --- | --- |
| E1, 2× A40, steps 1–30 | 52.7 s | 18.2 s | 26.7 s | 8.2 s |
| Colocated, 1× A40, steps 1–30 | 50.0 s | 16.4 s | 26.5 s | 7.1 s |

Training took the same time on a shared GPU, and rollout and weight sync were slightly faster.
The colocated run used a different host, so the 2.7 s difference shows only that colocation is not slower, not that it is faster.
Step 1 took 355 s because the vLLM engine starts at the first rollout when there is no step-0 eval; it does not affect the median.

Training metrics matched E1's first 30 steps: mean reward 0.484 against 0.480, and median sampler/trainer mismatch KL 0.0004 in both.

Memory was not a constraint.
vLLM's KV cache held 1.83 million tokens, while a step needs at most 256 sequences of 1,024 tokens (262,144 tokens).
GPU memory peaked at 40.4 of 46 GB.

## Limitations

The run has one seed and 30 steps, on a different host from E1.
Async rollouts were not tested colocated: with both phases on one GPU they would compete for the same compute, so E2's overlap may not carry over.

## Appendix: Reproducibility

Using the `COMMON` flags of the [E1 report](../2026-11-06-e0-e1-grpo-gsm8k/e1-report.md#appendix-reproducibility), with `--gpus-per-worker 0.5 --gpu-memory-utilization 0.5`:

```bash
python -m vrollout.train "${COMMON[@]}" \
  --num-iterations 30 \
  --lr-warmup-steps 10 \
  --eval-every 0 \
  --wandb-run-name e1-colocated-1xa40-30
```

Later commits also reset vLLM's prefix cache after every weight update, so a rerun on them is not bit-comparable with E1.

The training history is saved as [colocated-history.csv](./colocated-history.csv); the phase medians come from the W&B history export:

```bash
uv run scripts/report_stats.py step-times runs/history-e1-colocated.json
```
