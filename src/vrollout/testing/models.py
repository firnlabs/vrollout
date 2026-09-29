"""Models for testing."""

from torch import nn
from torch import Tensor


class ToyBlock(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4 * d_model, d_model)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.fc2(self.act(self.fc1(self.ln(x))))


class ToyModel(nn.Module):
    def __init__(self, *, d_model: int, num_layers: int, vocab_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([ToyBlock(d_model) for _ in range(num_layers)])
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor | None = None
    ) -> Tensor:
        _ = attention_mask  # unused: kept for API compatiblity
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


TEST_MODEL_SPECS = {
    "test-model": {"d_model": 16, "num_layers": 2, "vocab_size": 32},
}


def create_model(model_name: str):
    if model_name not in TEST_MODEL_SPECS:
        raise ValueError(f"Unsupported model: {model_name}")

    model_kwargs = TEST_MODEL_SPECS[model_name]
    return ToyModel(**model_kwargs)
