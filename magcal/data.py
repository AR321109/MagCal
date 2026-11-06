"""Input validation and MATLAB scene loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat


@dataclass(frozen=True)
class Scene:
    """A hyperspectral cube and optional evaluation-only ground truth."""

    name: str
    cube: np.ndarray
    gt: np.ndarray | None


def validate_cube(cube: np.ndarray) -> np.ndarray:
    cube = np.asarray(cube, dtype=np.float64)
    if cube.ndim != 3:
        raise ValueError(f"expected cube with shape (H,W,B), got {cube.shape}")
    if min(cube.shape) < 2:
        raise ValueError(f"degenerate cube shape: {cube.shape}")
    if not np.isfinite(cube).all():
        raise ValueError("cube contains NaN or infinite values")
    return cube


def validate_gt(gt: np.ndarray, spatial_shape: tuple[int, int]) -> np.ndarray:
    gt = np.asarray(gt).squeeze().astype(bool)
    if gt.shape != spatial_shape:
        raise ValueError(f"GT shape {gt.shape} does not match {spatial_shape}")
    if not gt.any() or gt.all():
        raise ValueError("GT must contain both background and anomaly pixels")
    return gt


def load_mat_scene(
    path: str | Path,
    data_key: str = "data",
    gt_key: str | None = "map",
) -> Scene:
    """Load one `.mat` scene; GT is optional and never used by the detector."""

    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    content = loadmat(path)
    if data_key not in content:
        raise KeyError(f"{path} has no {data_key!r}; keys={sorted(content)}")
    cube = validate_cube(content[data_key])
    gt = None
    if gt_key and gt_key in content:
        gt = validate_gt(content[gt_key], cube.shape[:2])
    return Scene(name=path.stem, cube=cube, gt=gt)
