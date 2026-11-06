"""Apply the paper's matched-kernel rarity control to a saved host map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from magcal.data import load_mat_scene
from magcal.evaluation import compare_scores
from magcal.magnitude import calibrate_score, patchwise_same_kernel_rarity
from magcal.preprocessing import pca_embedding, select_opbs, zscore_bands


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="input HSI .mat")
    parser.add_argument("--maps", required=True, type=Path, help="maps.npz from run.py")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data-key", default="data")
    parser.add_argument("--gt-key", default="map")
    parser.add_argument("--bands", type=int, default=64)
    parser.add_argument("--pca-components", type=int, default=5)
    parser.add_argument("--core", type=int, default=6)
    parser.add_argument("--halo", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()

    scene = load_mat_scene(args.input, args.data_key, args.gt_key or None)
    selected = select_opbs(scene.cube, args.bands)
    standardized = zscore_bands(scene.cube[:, :, selected])
    features = pca_embedding(standardized, args.pca_components)
    rarity = patchwise_same_kernel_rarity(features, core=args.core, halo=args.halo)

    with np.load(args.maps) as archive:
        if "original" not in archive:
            raise KeyError("maps archive must contain 'original'")
        original = archive["original"].astype(np.float64)
    calibrated = calibrate_score(original, rarity, args.alpha)

    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / "same_kernel_rarity.npz",
        original=original,
        rarity_prior=rarity,
        calibrated=calibrated,
    )
    summary = {
        "input": str(args.input.resolve()),
        "maps": str(args.maps.resolve()),
        "ground_truth_used_for_prior": False,
        "parameters": vars(args) | {"input": str(args.input), "maps": str(args.maps), "output": str(args.output)},
    }
    if scene.gt is not None:
        summary["metrics"] = compare_scores(scene.gt, original, calibrated)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(args.output / "summary.json")


if __name__ == "__main__":
    main()
