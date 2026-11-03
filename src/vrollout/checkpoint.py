"""Full-model checkpointing for plain and FSDP2-sharded actors."""

from pathlib import Path

import torch
from torch import nn
from torch.distributed.checkpoint import state_dict as dcp_state_dict

from vrollout.types import Rank

STATE_DICT_FILENAME = "model.pt"


def should_checkpoint(step: int, every: int, num_iterations: int) -> bool:
    """Returns whether to save after 1-indexed `step`: every N steps, and the last.

    Args:
        step: The 1-indexed training iteration that just finished.
        every: Save every `every` iterations; 0 saves only after the last one.
        num_iterations: Total number of training iterations.

    Returns:
        True if a checkpoint should be saved after `step`.
    """
    return step == num_iterations or (every > 0 and step % every == 0)


def save_checkpoint(model: nn.Module, path: Path, rank: Rank) -> Path | None:
    """Gathers the full model state on rank 0 and saves it to `path`.

    Must be called on every rank: gathering FSDP2 shards is a collective. Hugging
    Face models are saved with `save_pretrained` so the directory can be loaded by
    `from_pretrained` or vLLM; other models are saved as a `torch.save` state dict.
    Creating the directory raises `FileExistsError` if `path` is an existing file.

    Args:
        model: The model to save, sharded with FSDP2 or not.
        path: Directory to write the checkpoint into.
        rank: This process's rank; only rank 0 writes.

    Returns:
        `path` on rank 0, None on other ranks.
    """
    # full_state_dict gathers shards; cpu_offload keeps them only on rank 0
    state_dict = dcp_state_dict.get_model_state_dict(
        model,
        options=dcp_state_dict.StateDictOptions(full_state_dict=True, cpu_offload=True),
    )
    if rank != 0:
        return None

    path.mkdir(parents=True, exist_ok=True)
    save_pretrained = getattr(model, "save_pretrained", None)
    if callable(save_pretrained):
        save_pretrained(path, state_dict=state_dict)
    else:
        torch.save(state_dict, path / STATE_DICT_FILENAME)
    return path
