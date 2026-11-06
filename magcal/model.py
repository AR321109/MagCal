"""Minimal MSNet architecture used as the host residual detector."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.functional import interpolate


class Bottleneck(nn.Module):
    def __init__(self, channels: int, hidden_channels: int = 16):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, channels, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x) + x


class Encoder(nn.Module):
    def __init__(self, shape: tuple[int, int, int], down_rate: int):
        super().__init__()
        rows, cols, channels = shape
        self.output_size = (rows, cols)
        self.bottleneck = Bottleneck(channels)
        self.pool = nn.Sequential(nn.ReLU(), nn.AvgPool2d(down_rate))
        self.project = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.pool(self.bottleneck(x))
        skip = interpolate(encoded, size=self.output_size, mode="bilinear")
        return encoded, self.project(skip)


class Decoder(nn.Module):
    def __init__(self, shape: tuple[int, int, int]):
        super().__init__()
        rows, cols, channels = shape
        self.output_size = (rows, cols)
        self.pre = nn.Conv2d(channels, channels, 1)
        self.relu = nn.ReLU()
        self.post = nn.Conv2d(channels, channels, 1)

    def forward(self, encoded: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        decoded = interpolate(encoded, size=self.output_size, mode="bilinear")
        decoded = self.relu(self.pre(decoded))
        return self.post(decoded + skip)


class MSNet(nn.Module):
    def __init__(self, shape: tuple[int, int, int], layers: int = 3):
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be positive")
        self.layers = layers
        self.encoders = nn.ModuleList(
            Encoder(shape, down_rate=2**index) for index in range(layers)
        )
        self.decoders = nn.ModuleList(Decoder(shape) for _ in range(layers))

    def forward(self, cube: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x = cube.permute(2, 0, 1).unsqueeze(0)
        encoded_outputs: list[torch.Tensor] = []
        encoded_sum = x
        for encoder in self.encoders:
            encoded, skip = encoder(encoded_sum)
            encoded_outputs.append(encoded)
            encoded_sum = encoded_sum + skip

        decoded_sum = torch.zeros_like(x)
        decoded_outputs: list[torch.Tensor] = [torch.empty(0)] * self.layers
        for index in range(self.layers - 1, -1, -1):
            decoded = self.decoders[index](encoded_outputs[index], decoded_sum)
            decoded_sum = decoded_sum + decoded
            decoded_outputs[index] = decoded.squeeze(0).permute(1, 2, 0)
        return tuple(decoded_outputs)
