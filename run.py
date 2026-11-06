"""Run the compact MSNet + Magnitude calibration pipeline."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from magcal import MagCalConfig, MagCalDetector
from magcal.data import load_mat_scene
from magcal.evaluation import compare_scores
from magcal.visualization import save_candidate_changes, save_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="input .mat scene")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data-key", default="data")
    parser.add_argument("--gt-key", default="map", help="empty string disables GT loading")
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--bands", type=int, default=64)
    parser.add_argument("--pca-components", type=int, default=5)
    parser.add_argument("--core", type=int, default=6)
    parser.add_argument("--halo", type=int, default=1)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--separation-weight", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--output-iteration", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--top-fraction", type=float, default=0.05)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    scene = load_mat_scene(args.input, args.data_key, args.gt_key or None)

    config = MagCalConfig(
        selected_bands=args.bands,
        pca_components=args.pca_components,
        core=args.core,
        halo=args.halo,
        ridge=args.ridge,
        alpha=args.alpha,
        layers=args.layers,
        learning_rate=args.learning_rate,
        separation_weight=args.separation_weight,
        epochs=args.epochs,
        iterations=args.iterations,
        output_iteration=args.output_iteration,
        seed=args.seed,
        device=args.device,
        verbose=args.verbose,
    )

    print(f"Scene: {scene.name}, cube={scene.cube.shape}, GT={scene.gt is not None}")
    detector = MagCalDetector(config).fit(scene.cube)
    maps = detector.maps()

    archive = dict(maps)
    if scene.gt is not None:
        archive["gt"] = scene.gt
    np.savez_compressed(args.output / "maps.npz", **archive)

    metrics = None
    candidate_changes = None
    if scene.gt is not None:
        metrics = compare_scores(scene.gt, maps["original"], maps["calibrated"])
        write_json(args.output / "metrics.json", metrics)
        candidate_changes = save_candidate_changes(
            args.output / "candidate_changes.png",
            scene.gt,
            maps["original"],
            maps["calibrated"],
            fraction=args.top_fraction,
        )

    save_comparison(
        args.output / "comparison.png",
        scene.cube,
        scene.gt,
        maps["original"],
        maps["magnitude_prior"],
        maps["calibrated"],
        metrics,
    )

    run_record = {
        "scene": scene.name,
        "input": str(args.input.resolve()),
        "shape": list(scene.cube.shape),
        "ground_truth_used_for_detection": False,
        "config": asdict(config),
        "band_indices": detector.band_indices_.tolist(),
        "training_loss_by_iteration": detector.loss_history_,
        "magnitude_stats": detector.magnitude_stats_,
        "runtime_seconds": detector.runtime_seconds_,
        "candidate_changes": candidate_changes,
    }
    write_json(args.output / "run.json", run_record)

    print("Outputs:")
    for name in ("maps.npz", "run.json", "comparison.png"):
        print(f"  {args.output / name}")
    if metrics is not None:
        print(f"  {args.output / 'metrics.json'}")
        print(f"  {args.output / 'candidate_changes.png'}")
        for method in ("original", "calibrated"):
            values = metrics[method]
            print(
                f"{method:>10s}: AP={values['AP']:.5f}, "
                f"AUC={values['ROC_AUC']:.5f}, "
                f"FPR@95={values['FPR_at_95_TPR']:.5f}, "
                f"P@1%={values['precision_at_top_1pct']:.5f}"
            )


if __name__ == "__main__":
    main()
