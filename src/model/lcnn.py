import torch
from torch import nn
from torch.nn import Sequential


class MFMConv2d(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        kernel_size,
        stride=1,
        padding=0,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            input_channels,
            2 * output_channels,
            kernel_size,
            stride,
            padding,
        )

    def forward(self, input_data):
        first, second = torch.chunk(
            self.conv(input_data),
            chunks=2,
            dim=1,
        )

        return torch.maximum(first, second)


class MFMLinear(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
    ):
        super().__init__()

        self.linear = nn.Linear(
            input_size,
            2 * output_size,
        )

    def forward(self, input_data):
        first, second = torch.chunk(
            self.linear(input_data),
            chunks=2,
            dim=1,
        )

        return torch.maximum(first, second)


def initialize_lcnn_weights(module):
    if isinstance(
        module,
        (nn.Conv2d, nn.Linear),
    ):
        nn.init.kaiming_normal_(
            module.weight,
            mode="fan_out",
            nonlinearity="relu",
        )

        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(
        module,
        (nn.BatchNorm1d, nn.BatchNorm2d),
    ):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class LCNN(nn.Module):
    def __init__(
        self,
        dropout=0.75,
        num_classes=2,
    ):
        super().__init__()

        self.features = Sequential(
            MFMConv2d(
                1,
                32,
                kernel_size=5,
                padding=2,
            ),
            nn.MaxPool2d(2, 2),
            MFMConv2d(
                32,
                32,
                kernel_size=1,
            ),
            nn.BatchNorm2d(32),
            MFMConv2d(
                32,
                48,
                kernel_size=3,
                padding=1,
            ),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(48),
            MFMConv2d(
                48,
                48,
                kernel_size=1,
            ),
            nn.BatchNorm2d(48),
            MFMConv2d(
                48,
                64,
                kernel_size=3,
                padding=1,
            ),
            nn.MaxPool2d(2, 2),
            MFMConv2d(
                64,
                64,
                kernel_size=1,
            ),
            nn.BatchNorm2d(64),
            MFMConv2d(
                64,
                32,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            MFMConv2d(
                32,
                32,
                kernel_size=1,
            ),
            nn.BatchNorm2d(32),
            MFMConv2d(
                32,
                32,
                kernel_size=3,
                padding=1,
            ),
            nn.MaxPool2d(2, 2),
        )

        self.embedding = MFMLinear(
            32 * 1 * 37,
            80,
        )

        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(80)
        self.classifier = nn.Linear(
            80,
            num_classes,
        )

        self.apply(initialize_lcnn_weights)

    def forward(self, audio, **batch):
        features = audio.unsqueeze(1)
        features = self.features(features)
        features = torch.flatten(
            features,
            start_dim=1,
        )

        embedding = self.embedding(features)
        embedding = self.dropout(embedding)
        embedding = self.batch_norm(embedding)

        return {
            "logits": self.classifier(embedding),
        }

    def __str__(self):
        all_parameters = sum(
            parameter.numel()
            for parameter in self.parameters()
        )

        trainable_parameters = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

        return (
            f"{super().__str__()}"
            f"\nAll parameters: {all_parameters}"
            f"\nTrainable parameters: {trainable_parameters}"
        )
    