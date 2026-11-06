"""End-to-end MSNet residual plus Magnitude calibration pipeline."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

import numpy as np
import torch

from .data import validate_cube
from .magnitude import calibrate_score, patchwise_magnitude
from .preprocessing import minmax, pca_embedding, select_opbs, zscore_bands
from .training import TrainingConfig, train_msnet


@dataclass(frozen=True)
class MagCalConfig:
    selected_bands: int = 64
    pca_components: int = 5
    core: int = 6
    halo: int = 1
    ridge: float = 1e-3
    alpha: float = 1.0
    layers: int = 3
    learning_rate: float = 1e-3
    separation_weight: float = 1e-3
    epochs: int = 150
    iterations: int = 5
    output_iteration: int = 5
    seed: int = 20261020
    device: str = "auto"
    verbose: bool = False

    def validate(self, input_bands: int) -> None:
        if not 1 <= self.selected_bands <= input_bands:
            raise ValueError(
                f"selected_bands must be in [1,{input_bands}], "
                f"got {self.selected_bands}"
            )
        if not 1 <= self.pca_components <= self.selected_bands:
            raise ValueError("pca_components must not exceed selected_bands")
        if self.core < 1 or self.halo < 0 or self.ridge <= 0 or self.alpha < 0:
            raise ValueError("invalid Magnitude parameters")
        TrainingConfig(
            layers=self.layers,
            learning_rate=self.learning_rate,
            separation_weight=self.separation_weight,
            epochs=self.epochs,
            iterations=self.iterations,
            output_iteration=self.output_iteration,
            verbose=self.verbose,
        ).validate()


class MagCalDetector:
    """Fit MSNet without labels, then calibrate its residual with Magnitude."""

    def __init__(self, config: MagCalConfig | None = None):
        self.config = config or MagCalConfig()
        self.original_: np.ndarray | None = None
        self.prior_: np.ndarray | None = None
        self.calibrated_: np.ndarray | None = None
        self.band_indices_: np.ndarray | None = None
        self.loss_history_: list[float] = []
        self.magnitude_stats_: dict[str, float] = {}
        self.runtime_seconds_: dict[str, float] = {}

    def _device(self) -> torch.device:
        if self.config.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.config.device)

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def fit(self, cube: np.ndarray) -> "MagCalDetector":
        """Fit from an HSI cube only. No ground truth is accepted here."""

        cube = validate_cube(cube)
        self.config.validate(cube.shape[2])
        self._set_seed(self.config.seed)
        started = time.perf_counter()

        band_indices = select_opbs(cube, self.config.selected_bands)
        selected = zscore_bands(cube[:, :, band_indices])
        embedded = pca_embedding(selected, self.config.pca_components)
        preprocessed = time.perf_counter()

        prior, stats = patchwise_magnitude(
            embedded,
            core=self.config.core,
            halo=self.config.halo,
            ridge=self.config.ridge,
        )
        prior_done = time.perf_counter()

        training_config = TrainingConfig(
            layers=self.config.layers,
            learning_rate=self.config.learning_rate,
            separation_weight=self.config.separation_weight,
            epochs=self.config.epochs,
            iterations=self.config.iterations,
            output_iteration=self.config.output_iteration,
            verbose=self.config.verbose,
        )
        raw_residual, losses = train_msnet(selected, training_config, self._device())
        trained = time.perf_counter()
        original = minmax(raw_residual)
        calibrated = calibrate_score(raw_residual, prior, self.config.alpha)
        finished = time.perf_counter()

        self.original_ = original
        self.prior_ = prior
        self.calibrated_ = calibrated
        self.band_indices_ = band_indices
        self.loss_history_ = losses
        self.magnitude_stats_ = stats
        self.runtime_seconds_ = {
            "preprocessing": preprocessed - started,
            "magnitude_prior": prior_done - preprocessed,
            "msnet_training": trained - prior_done,
            "calibration": finished - trained,
            "total": finished - started,
        }
        return self

    def maps(self) -> dict[str, np.ndarray]:
        if self.original_ is None or self.prior_ is None or self.calibrated_ is None:
            raise RuntimeError("call fit(cube) before requesting maps")
        return {
            "original": self.original_,
            "magnitude_prior": self.prior_,
            "calibrated": self.calibrated_,
        }
