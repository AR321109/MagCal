"""Ground-truth-free MSNet separation training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn
from torch.optim import Adam
from tqdm import trange

from .model import MSNet


def residual_map(cube: torch.Tensor, outputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Squared spectral reconstruction error of the first decoder output."""

    return ((cube - outputs[0]).detach() ** 2).sum(dim=2)


def _minmax_tensor(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    span = x.max() - x.min()
    if float(span.detach().cpu()) <= eps:
        return torch.zeros_like(x)
    return (x - x.min()) / span


class SoftMask:
    def __init__(self, shape: tuple[int, int], device: torch.device):
        self.value = torch.zeros((*shape, 1), device=device)

    def update(self, score: torch.Tensor) -> None:
        self.value = _minmax_tensor(score).unsqueeze(-1)

    @property
    def background(self) -> torch.Tensor:
        return 1.0 - self.value


class LaplacianOfGaussian(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        kernel = torch.tensor(
            [
                [-2, -4, -4, -4, -2],
                [-4, 0, 8, 0, -4],
                [-4, 8, 24, 8, -4],
                [-4, 0, 8, 0, -4],
                [-2, -4, -4, -4, -2],
            ],
            dtype=torch.float32,
            device=device,
        )
        self.register_buffer("kernel", kernel.reshape(1, 1, 1, 5, 5))

    def forward(self, cube: torch.Tensor) -> torch.Tensor:
        x = cube.permute(2, 0, 1).unsqueeze(0)
        x = functional.pad(x, (2, 2, 2, 2), mode="reflect").unsqueeze(0)
        x = functional.conv3d(x, self.kernel)
        return x.squeeze(0).squeeze(0).permute(1, 2, 0)


class SeparationLoss(nn.Module):
    def __init__(self, weight: float, device: torch.device):
        super().__init__()
        self.weight = weight
        self.log = LaplacianOfGaussian(device)

    def forward(
        self,
        cube: torch.Tensor,
        outputs: tuple[torch.Tensor, ...],
        mask: SoftMask,
    ) -> torch.Tensor:
        reconstruction = outputs[0]
        anomaly_count = mask.value.sum()
        background_count = mask.background.sum()
        anomaly_loss = ((mask.value * self.log(reconstruction)) ** 2).sum()
        anomaly_loss /= anomaly_count + 1e-6
        background_loss = ((mask.background * (cube - reconstruction)) ** 2).sum()
        background_loss /= background_count + 1e-6
        return background_loss + self.weight * anomaly_loss


class MultiScaleReconstructionLoss(nn.Module):
    def forward(
        self, cube: torch.Tensor, outputs: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        losses = []
        scale = 1
        for output in outputs:
            rows = output.shape[0] // scale
            cols = output.shape[1] // scale
            target = functional.interpolate(
                cube.permute(2, 0, 1).unsqueeze(0),
                size=(rows, cols),
                mode="bilinear",
            )
            estimate = functional.interpolate(
                output.permute(2, 0, 1).unsqueeze(0),
                size=(rows, cols),
                mode="bilinear",
            )
            losses.append(((target - estimate) ** 2).sum() / (rows * cols))
            scale *= 2
        return sum(losses) / len(losses)


class TotalLoss(nn.Module):
    def __init__(self, separation_weight: float, device: torch.device):
        super().__init__()
        self.separation = SeparationLoss(separation_weight, device)
        self.multiscale = MultiScaleReconstructionLoss()

    def forward(
        self,
        cube: torch.Tensor,
        outputs: tuple[torch.Tensor, ...],
        mask: SoftMask,
    ) -> torch.Tensor:
        channels = cube.shape[2]
        return (
            self.separation(cube, outputs, mask)
            + self.multiscale(cube, outputs)
        ) / channels


@dataclass(frozen=True)
class TrainingConfig:
    layers: int = 3
    learning_rate: float = 1e-3
    separation_weight: float = 1e-3
    epochs: int = 150
    iterations: int = 5
    output_iteration: int = 5
    verbose: bool = False

    def validate(self) -> None:
        if self.layers < 1 or self.epochs < 1 or self.iterations < 1:
            raise ValueError("layers, epochs, and iterations must be positive")
        if not 1 <= self.output_iteration <= self.iterations:
            raise ValueError("output_iteration must lie within training iterations")
        if self.learning_rate <= 0 or self.separation_weight < 0:
            raise ValueError("invalid optimization parameters")


def train_msnet(
    cube: np.ndarray,
    config: TrainingConfig,
    device: torch.device,
) -> tuple[np.ndarray, list[float]]:
    """Train MSNet without labels and return the locked-iteration residual."""

    config.validate()
    rows, cols, channels = cube.shape
    cube_tensor = torch.from_numpy(cube).to(device=device, dtype=torch.float32)
    model = MSNet((rows, cols, channels), layers=config.layers).to(device)
    criterion = TotalLoss(config.separation_weight, device)
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    mask = SoftMask((rows, cols), device)
    iteration_losses: list[float] = []
    selected_score: np.ndarray | None = None

    for iteration in range(1, config.iterations + 1):
        epoch_range = trange(
            config.epochs,
            disable=not config.verbose,
            desc=f"SeT {iteration}/{config.iterations}",
            leave=False,
        )
        running_loss = 0.0
        model.train()
        for _ in epoch_range:
            optimizer.zero_grad()
            outputs = model(cube_tensor)
            loss = criterion(cube_tensor, outputs, mask)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().cpu())
            if config.verbose:
                epoch_range.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}")

        with torch.no_grad():
            outputs = model(cube_tensor)
            score = residual_map(cube_tensor, outputs)
            mask.update(score)
        iteration_losses.append(running_loss / config.epochs)
        if iteration == config.output_iteration:
            selected_score = score.cpu().numpy().astype(np.float64)

    if selected_score is None:
        raise RuntimeError("training finished without selecting an output map")
    return selected_score, iteration_losses
