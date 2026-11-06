"""Patchwise spatial--spectral Magnitude and bounded score calibration."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from .preprocessing import minmax


def _positive_median(distance: np.ndarray) -> float:
    values = distance[distance > 0]
    return float(np.median(values)) if values.size else 1.0


def _solve_patch(
    features: np.ndarray,
    spatial_weight: float,
    spectral_weight: float,
    ridge: float,
) -> tuple[np.ndarray, float]:
    height, width, _ = features.shape
    yy, xx = np.mgrid[0:height, 0:width]
    coordinates = np.column_stack((yy.ravel(), xx.ravel())).astype(np.float64)
    spectra = features.reshape(height * width, -1).astype(np.float64)

    d_spatial = cdist(coordinates, coordinates, metric="cityblock")
    d_spectral = cdist(spectra, spectra, metric="cityblock")
    distance = spatial_weight * d_spatial / max(_positive_median(d_spatial), 1e-12)
    distance += spectral_weight * d_spectral / max(_positive_median(d_spectral), 1e-12)

    similarity = np.exp(-distance)
    system = similarity + ridge * np.eye(similarity.shape[0])
    weights = np.linalg.solve(system, np.ones(system.shape[0]))
    return np.abs(weights).reshape(height, width), float(np.linalg.cond(system))


def _same_kernel_rarity_patch(
    features: np.ndarray,
    spatial_weight: float,
    spectral_weight: float,
) -> np.ndarray:
    """Compute ``1 - mean off-diagonal similarity`` from MagCal's kernel."""

    height, width, _ = features.shape
    yy, xx = np.mgrid[0:height, 0:width]
    coordinates = np.column_stack((yy.ravel(), xx.ravel())).astype(np.float64)
    spectra = features.reshape(height * width, -1).astype(np.float64)
    d_spatial = cdist(coordinates, coordinates, metric="cityblock")
    d_spectral = cdist(spectra, spectra, metric="cityblock")
    distance = spatial_weight * d_spatial / max(_positive_median(d_spatial), 1e-12)
    distance += spectral_weight * d_spectral / max(_positive_median(d_spectral), 1e-12)
    similarity = np.exp(-distance)
    count = similarity.shape[0]
    mean_off_diagonal = (similarity.sum(axis=1) - 1.0) / max(count - 1, 1)
    return (1.0 - mean_off_diagonal).reshape(height, width)


def _validate_patch_parameters(
    features: np.ndarray,
    core: int,
    halo: int,
    spatial_weight: float,
    spectral_weight: float,
) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 3:
        raise ValueError(f"features must be (H,W,C), got {features.shape}")
    if core < 1 or halo < 0:
        raise ValueError("require core>=1 and halo>=0")
    if spatial_weight < 0 or spectral_weight < 0:
        raise ValueError("distance weights must be nonnegative")
    if spatial_weight == 0 and spectral_weight == 0:
        raise ValueError("at least one distance weight must be positive")
    return features


def _padded_features(features: np.ndarray, core: int, halo: int):
    rows, cols, _ = features.shape
    target_rows = int(np.ceil(rows / core) * core)
    target_cols = int(np.ceil(cols / core) * core)
    padded = np.pad(
        features,
        ((halo, halo + target_rows - rows), (halo, halo + target_cols - cols), (0, 0)),
        mode="reflect",
    )
    return padded, target_rows, target_cols


def patchwise_magnitude(
    features: np.ndarray,
    core: int = 6,
    halo: int = 1,
    spatial_weight: float = 1.0,
    spectral_weight: float = 1.0,
    ridge: float = 1e-3,
) -> tuple[np.ndarray, dict[str, float]]:
    """Compute one local Magnitude weight per image pixel.

    The context halo overlaps, but only the non-overlapping core is written
    back. This preserves local context without averaging incompatible systems.
    """

    features = _validate_patch_parameters(
        features, core, halo, spatial_weight, spectral_weight
    )
    if ridge <= 0:
        raise ValueError("ridge must be positive")

    rows, cols, channels = features.shape
    padded, target_rows, target_cols = _padded_features(features, core, halo)

    prior = np.zeros((target_rows, target_cols), dtype=np.float64)
    conditions: list[float] = []
    for top in range(0, target_rows, core):
        for left in range(0, target_cols, core):
            patch = padded[
                top : top + core + 2 * halo,
                left : left + core + 2 * halo,
                :,
            ]
            magnitude, condition = _solve_patch(
                patch, spatial_weight, spectral_weight, ridge
            )
            prior[top : top + core, left : left + core] = magnitude[
                halo : halo + core, halo : halo + core
            ]
            conditions.append(condition)

    cropped = minmax(prior[:rows, :cols])
    stats = {
        "condition_median": float(np.median(conditions)),
        "condition_max": float(np.max(conditions)),
        "patch_count": float(len(conditions)),
        "patch_pixels": float((core + 2 * halo) ** 2),
        "pca_channels": float(channels),
    }
    return cropped, stats


def patchwise_same_kernel_rarity(
    features: np.ndarray,
    core: int = 6,
    halo: int = 1,
    spatial_weight: float = 1.0,
    spectral_weight: float = 1.0,
) -> np.ndarray:
    """Matched control that changes only the kernel-to-prior aggregation.

    PCA features, distance scaling, similarity kernel, padding, core stitching,
    scene normalization, host score and calibration equation remain identical
    to MagCal.  No Magnitude linear system is solved.
    """

    features = _validate_patch_parameters(
        features, core, halo, spatial_weight, spectral_weight
    )
    rows, cols, _ = features.shape
    padded, target_rows, target_cols = _padded_features(features, core, halo)
    prior = np.zeros((target_rows, target_cols), dtype=np.float64)
    for top in range(0, target_rows, core):
        for left in range(0, target_cols, core):
            patch = padded[
                top : top + core + 2 * halo,
                left : left + core + 2 * halo,
                :,
            ]
            rarity = _same_kernel_rarity_patch(
                patch, spatial_weight, spectral_weight
            )
            prior[top : top + core, left : left + core] = rarity[
                halo : halo + core, halo : halo + core
            ]
    return minmax(prior[:rows, :cols])


def calibrate_score(score: np.ndarray, prior: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Apply the frozen protocol ``R=N(E); S=N(R*(1+alpha*M))``.

    ``score`` may be an unnormalized nonnegative host residual ``E``.  The
    mandatory first min--max operation reproduces EXP-010 and is not omitted
    even though a second min--max operation follows the multiplicative rule.
    """

    score = np.asarray(score, dtype=np.float64)
    prior = np.asarray(prior, dtype=np.float64)
    if score.shape != prior.shape:
        raise ValueError(f"score {score.shape} and prior {prior.shape} differ")
    if alpha < 0:
        raise ValueError("alpha must be nonnegative")
    if np.any(score < 0) or not np.isfinite(score).all():
        raise ValueError("host score must be finite and nonnegative")
    if np.any((prior < 0) | (prior > 1)) or not np.isfinite(prior).all():
        raise ValueError("prior must be finite and normalized to [0,1]")
    normalized_score = minmax(score)
    return minmax(normalized_score * (1.0 + alpha * prior))
