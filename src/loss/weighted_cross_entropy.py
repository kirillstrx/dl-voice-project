import torch
from torch import nn


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss = nn.CrossEntropyLoss(
            weight=torch.tensor(
                [1.0, 8.84],
                dtype=torch.float32,
            )
        )

    def forward(
        self,
        logits,
        labels,
        **batch,
    ):
        return {
            "loss": self.loss(logits, labels),
        }
    