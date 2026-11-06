"""Shared preprocessing for the host detector and Magnitude branch."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


def minmax(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    lo = float(np.nanmin(x))
    hi = float(np.nanmax(x))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo <= eps:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def zscore_bands(cube: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mean = cube.mean(axis=(0, 1), keepdims=True)
    std = cube.std(axis=(0, 1), keepdims=True)
    return (cube - mean) / np.maximum(std, eps)


def select_opbs(cube: np.ndarray, count: int) -> np.ndarray:
    """Orthogonal-projection-based band selection used by MSNet."""

    rows, cols, bands = cube.shape
    if not 1 <= count <= bands:
        raise ValueError(f"bands must be in [1,{bands}], got {count}")
    flat = cube.reshape(rows * cols, bands).astype(np.float64)
    residual = flat.copy()
    energy = np.zeros(bands, dtype=np.float64)
    chosen = [int(np.argmax(np.var(flat, axis=0)))]
    energy[chosen[0]] = float(np.dot(residual[:, chosen[0]], residual[:, chosen[0]]))

    while len(chosen) < count:
        previous = chosen[-1]
        denominator = energy[previous] + 1e-9
        best_energy, best_band = -np.inf, -1
        for band in range(bands):
            if band in chosen:
                continue
            residual[:, band] -= residual[:, previous] * (
                np.dot(residual[:, previous], residual[:, band]) / denominator
            )
            energy[band] = float(np.dot(residual[:, band], residual[:, band]))
            if energy[band] > best_energy:
                best_energy, best_band = energy[band], band
        chosen.append(best_band)
    return np.asarray(sorted(chosen), dtype=np.int64)


def pca_embedding(cube: np.ndarray, components: int) -> np.ndarray:
    rows, cols, bands = cube.shape
    if not 1 <= components <= bands:
        raise ValueError(f"PCA components must be in [1,{bands}]")
    flat = cube.reshape(-1, bands)
    embedded = PCA(n_components=components, svd_solver="full").fit_transform(flat)
    embedded /= np.maximum(embedded.std(axis=0, keepdims=True), 1e-12)
    return embedded.reshape(rows, cols, components)
