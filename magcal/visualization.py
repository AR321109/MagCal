"""Visualizations designed to expose Magnitude's ranking effect."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from sklearn.decomposition import PCA

from .evaluation import percentile_rank
from .preprocessing import minmax


def pseudo_rgb(cube: np.ndarray) -> np.ndarray:
    flat = cube.reshape(-1, cube.shape[2])
    components = min(3, cube.shape[2])
    rgb = PCA(n_components=components).fit_transform(flat)
    if components < 3:
        rgb = np.pad(rgb, ((0, 0), (0, 3 - components)))
    rgb = rgb.reshape(*cube.shape[:2], 3)
    for channel in range(3):
        lo, hi = np.percentile(rgb[:, :, channel], [1, 99])
        rgb[:, :, channel] = np.clip((rgb[:, :, channel] - lo) / (hi - lo + 1e-12), 0, 1)
    return rgb


def save_comparison(
    output: str | Path,
    cube: np.ndarray,
    gt: np.ndarray | None,
    original: np.ndarray,
    prior: np.ndarray,
    calibrated: np.ndarray,
    metrics: dict[str, dict[str, float]] | None = None,
) -> None:
    """Save the host, prior, calibrated map, and explicit rank displacement."""

    original_view = minmax(original)
    calibrated_view = minmax(calibrated)
    rank_gain = percentile_rank(calibrated) - percentile_rank(original)
    limit = max(float(np.max(np.abs(rank_gain))), 1e-6)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes = axes.ravel()
    panels = [
        (pseudo_rgb(cube), "Pseudo-RGB", None, {}),
        (gt if gt is not None else np.zeros(cube.shape[:2]), "Ground truth", "gray", {"vmin": 0, "vmax": 1}),
        (original_view, "Original MSNet", "magma", {"vmin": 0, "vmax": 1}),
        (prior, "Local Magnitude prior", "viridis", {"vmin": 0, "vmax": 1}),
        (calibrated_view, "MSNet + Magnitude", "magma", {"vmin": 0, "vmax": 1}),
        (rank_gain, "Percentile-rank change", "coolwarm", {"vmin": -limit, "vmax": limit}),
    ]
    if metrics:
        panels[2] = (
            panels[2][0],
            f"Original MSNet\nAP={metrics['original']['AP']:.4f}",
            panels[2][2],
            panels[2][3],
        )
        panels[4] = (
            panels[4][0],
            f"MSNet + Magnitude\nAP={metrics['calibrated']['AP']:.4f}",
            panels[4][2],
            panels[4][3],
        )

    for axis, (image, title, cmap, kwargs) in zip(axes, panels):
        shown = axis.imshow(image, cmap=cmap, **kwargs)
        axis.set_title(title)
        axis.axis("off")
        if cmap is not None and title != "Ground truth":
            fig.colorbar(shown, ax=axis, fraction=0.046, pad=0.04)
    if gt is None:
        axes[1].text(0.5, 0.5, "GT not supplied", ha="center", va="center", transform=axes[1].transAxes)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _top_mask(score: np.ndarray, fraction: float) -> np.ndarray:
    if not 0 < fraction < 1:
        raise ValueError("top fraction must lie in (0,1)")
    flat = np.asarray(score).ravel()
    count = max(1, int(np.ceil(fraction * flat.size)))
    indices = np.argpartition(flat, -count)[-count:]
    mask = np.zeros(flat.size, dtype=bool)
    mask[indices] = True
    return mask.reshape(score.shape)


def save_candidate_changes(
    output: str | Path,
    gt: np.ndarray,
    original: np.ndarray,
    calibrated: np.ndarray,
    fraction: float = 0.05,
) -> dict[str, int]:
    """Color the exact pixels entering or leaving a fixed candidate budget."""

    gt = np.asarray(gt, dtype=bool)
    original_top = _top_mask(original, fraction)
    calibrated_top = _top_mask(calibrated, fraction)
    recovered = gt & calibrated_top & ~original_top
    lost = gt & original_top & ~calibrated_top
    removed_fp = ~gt & original_top & ~calibrated_top
    added_fp = ~gt & calibrated_top & ~original_top
    stable_tp = gt & original_top & calibrated_top
    stable_fp = ~gt & original_top & calibrated_top

    image = np.zeros((*gt.shape, 3), dtype=np.float64)
    image[stable_fp] = (0.35, 0.35, 0.35)
    image[stable_tp] = (1.0, 1.0, 1.0)
    image[recovered] = (0.10, 0.85, 0.20)
    image[lost] = (1.00, 0.60, 0.05)
    image[removed_fp] = (0.10, 0.45, 1.00)
    image[added_fp] = (0.95, 0.10, 0.15)

    counts = {
        "recovered_anomalies": int(recovered.sum()),
        "lost_anomalies": int(lost.sum()),
        "removed_false_positives": int(removed_fp.sum()),
        "added_false_positives": int(added_fp.sum()),
        "stable_true_positives": int(stable_tp.sum()),
        "stable_false_positives": int(stable_fp.sum()),
    }
    fig, axis = plt.subplots(figsize=(7, 6))
    axis.imshow(image)
    axis.set_title(f"Top {100*fraction:.1f}% candidate changes")
    axis.axis("off")
    axis.legend(
        handles=[
            Patch(color=(0.10, 0.85, 0.20), label="Recovered anomaly"),
            Patch(color=(1.00, 0.60, 0.05), label="Lost anomaly"),
            Patch(color=(0.10, 0.45, 1.00), label="Removed background FP"),
            Patch(color=(0.95, 0.10, 0.15), label="Added background FP"),
            Patch(color="white", label="Stable anomaly"),
            Patch(color=(0.35, 0.35, 0.35), label="Stable background FP"),
        ],
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=True,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return counts
