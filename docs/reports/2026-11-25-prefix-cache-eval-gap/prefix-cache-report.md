# Prefix-cache eval gap

**Date:** November 25, 2026

## Summary

At step 200, both the [E1](../2026-11-06-e0-e1-grpo-gsm8k/e1-report.md) and [E2](../2026-11-18-e2-async-grpo-gsm8k/e2-report.md) runs' in-training eval scored below a separate eval of the saved checkpoint: E1 0.528 against 0.558, and E2 0.522 against 0.540.
The cause is vLLM's prefix cache, which survives a weight reload, so prompts whose prefix is cached reuse KV values computed by older weights.
Reloading E1's step-200 weights into an engine that had already evaluated the base model scored 0.528, the in-training score.
After `reset_prefix_cache()` the same engine scored 0.565 (+3.7 points, p = 0.001), close to a fresh load of the checkpoint (0.555).
E2's checkpoint showed the same direction, but its smaller gap was below what the test can resolve.
The fix is to reset the cache after every weight update, which also affects training rollouts.

## Background

The in-training eval runs in a vLLM engine whose weights are reloaded in place after every training step, while the separate eval loads the checkpoint into a new engine.
vLLM's prefix cache stores the KV values of each full 16-token prompt block and reuses them for later prompts that start with the same tokens.
`reload_weights` does not clear it, and vrollout never did either.
Every prompt starts with the same 24-token chat-template header, so one cached block is shared by all rollouts and evals, and the 500 eval prompts repeat every 25 steps, so their own blocks may also be served from the cache.

## Method

[prefix_cache_eval_gap.py](./prefix_cache_eval_gap.py) evaluates the first 500 GSM8K test problems with E1's eval settings (greedy decoding, 512-token budget, strict scoring) under four conditions, in this order:

| Condition | Engine state | Stands for |
| --- | --- | --- |
| `base` | New engine on the base model; fills the prefix cache | E0 |
| `inplace` (A) | The same engine after reloading the step-200 weights in place, as training does | In-training eval |
| `reset` (A′) | The same engine after `reset_prefix_cache()` | In-training eval with the fix |
| `fresh` (B) | New engine that loads the step-200 checkpoint from disk | Separate eval |

Each checkpoint was tested in 3 repetitions with new engines, because repeated greedy evals in vLLM already disagree on some problems.
[pooled_reps.py](./pooled_reps.py) averages each problem's correctness over the repetitions and compares conditions with a paired permutation test.
The decision rule, fixed before the run, was: if A < A′ ≈ B, reset the prefix cache after each weight update.
The test ran on 1× A40 with the E1 and E2 checkpoints and software.

## Results

Mean strict accuracy over 3 repetitions, and pooled paired differences (95% CI, permutation p):

| Checkpoint | inplace (A) | reset (A′) | fresh (B) | A′ − A | B − A′ |
| --- | --- | --- | --- | --- | --- |
| E1 step 200 | 0.528 | 0.565 | 0.555 | +3.7 ± 2.2 (p = 0.001) | −0.9 ± 1.5 (p = 0.24) |
| E2 step 200 | 0.519 | 0.526 | 0.536 | +0.7 ± 2.1 (p = 0.53) | +1.0 ± 1.4 (p = 0.18) |

For E1, A < A′ ≈ B: the in-place reload reproduces the in-training score, it scored lowest in each of the 3 repetitions, and resetting the cache recovers the fresh-load accuracy.
For E2, the in-place reload also scored lowest on average, but no difference is significant.
Its original gap was also smaller (1.8 against 3.0 points), and the test cannot resolve effects below about 2 points, so E2 is consistent with E1 but not evidence on its own.

The repetitions also size the noise of a clean eval: the same condition in two new engines flipped 5–9% of verdicts.
The E1 and E2 reports attributed the 8–11% of flips between their in-training and separate evals to vLLM's kernels not being batch-invariant.
Part of those flips came from stale KV.

## Fix

The reset is needed for correctness regardless of the accuracy effect.
Without it, cached blocks hold KV values computed by older weights, so the engine mixes two weight versions in one forward pass.
E1 shows that this can cost 3.7 points of accuracy.
The fix makes `RolloutWorker.update_weights` call `reset_prefix_cache()` after every weight reload.
It changes training as well as evaluation: without the reset, the shared header block was computed by the base weights and most likely served from the cache to every rollout of all 200 steps, so the sampler saw a slightly different context than the trainer.
E1's measured sampler/trainer mismatch was small (median KL 0.0004 per token), so the effect on training was probably small, but it was not measured separately.
Runs with the fix are not bit-comparable with E1 and E2.

Most other RL training frameworks do the same, as of this writing.
TRL's GRPO trainer, the closest to this project, loads new weights into the running vLLM engine and then calls `reset_prefix_cache()`; verl, OpenRLHF, NeMo-RL, AReaL, slime and ROLL also reset the cache at each weight sync.
In fully asynchronous training, prime-rl instead adds the policy version to each request's cache key, so requests still generating keep their cache while new ones never match older weights.
vrollout would need this only if it updated weights while a batch is still generating.

## Limitations

Only one of the two checkpoints shows a significant effect, and 3 repetitions are not enough to resolve E2's smaller difference.
The test reloads step-200 weights over a base-model cache, the largest possible weight gap.
During training, eval prompts' cached blocks were up to 25 steps old.

The effect on training was not measured: E1's mismatch KL (0.0004) includes both the stale header block and the sampler/trainer precision gap, and cannot separate them.
A short E1 run with the fix would be useful.
A lower mismatch KL would show the header's share, and in-training and separate evals that agree within clean-eval noise would confirm the fix closes the gap during training.

## Appendix: Reproducibility

[prefix_cache_eval_gap.py](./prefix_cache_eval_gap.py) is the script that produced the results.
It calls `reset_prefix_cache()` on the rollout manager for the `reset` condition.

With the fix, `RolloutWorker.update_weights` resets the cache after every reload, so the script's `inplace` condition is no longer stale and measures the fixed in-training eval instead.

To reproduce the stale results:

1. In `src/vrollout/generation.py`, delete the `await self._llm_engine.reset_prefix_cache()` line that follows the comment `# a reload keeps the prefix cache` in `RolloutWorker.update_weights`.
   This removes the fix.
2. Run the commands below for each checkpoint (a `step-200` directory saved by the E1 or E2 training command with `--checkpoint-dir`).
3. Restore the fix with `git checkout -- src/vrollout/generation.py`.

To measure the fixed condition instead, skip steps 1 and 3; `inplace` should then match `reset` (not run for this report).

```bash
uv run python docs/reports/2026-11-25-prefix-cache-eval-gap/prefix_cache_eval_gap.py \
  --checkpoint checkpoints/e1/step-200 \
  --output-dir eval/prefix-cache/e1-step-200 \
  --reps 3
uv run python docs/reports/2026-11-25-prefix-cache-eval-gap/pooled_reps.py \
  eval/prefix-cache/e1-step-200
```
