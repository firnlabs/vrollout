"""FSDP2 sharding helpers for actor and reference models."""

from torch import nn
from torch.distributed import fsdp
from torch.distributed.device_mesh import DeviceMesh

# attribute paths to the repeated transformer blocks of common model families
_BLOCK_PATHS: tuple[str, ...] = (
    "model.layers",  # Qwen2, Llama, Mistral
    "transformer.h",  # GPT-2
    "layers",  # vrollout.testing.ToyModel
)


def find_transformer_blocks(model: nn.Module) -> list[nn.Module]:
    """Returns the repeated transformer blocks of `model`, or [] if none are found."""
    for path in _BLOCK_PATHS:
        module: object = model
        for attr in path.split("."):
            module = getattr(module, attr, None)
            if module is None:
                break
        if isinstance(module, nn.ModuleList) and len(module) > 0:
            return list(module)
    return []


def shard_model(
    model: nn.Module,
    mesh: DeviceMesh,
    reshard_after_forward: bool = True,
) -> nn.Module:
    """Applies FSDP2 `fully_shard` per transformer block, then to the root module.

    Sharding each block separately lets FSDP2 all-gather one block's parameters at a
    time during forward/backward instead of materializing the full model. Parameters not
    inside a block (embeddings, final norm, tied LM head) are owned by the root.

    Must be called before the optimizer is constructed, because `fully_shard` replaces
    parameters with `DTensor` shards.
    """
    for block in find_transformer_blocks(model):
        fsdp.fully_shard(block, mesh=mesh, reshard_after_forward=reshard_after_forward)
    fsdp.fully_shard(model, mesh=mesh, reshard_after_forward=reshard_after_forward)
    return model
