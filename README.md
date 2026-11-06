# MagCal: Metric-Space Magnitude Calibration for Hyperspectral Anomaly Detection

This repository provides the compact reference implementation used to support
the paper **"Calibrating Reconstruction Residuals via Metric Space Magnitude
for Hyperspectral Anomaly Detection."**

MagCal leaves the host detector unchanged. It builds a local spectral--spatial
prior from regularized pointwise Magnitude weights and applies bounded
multiplicative reweighting to a normalized host score:

```text
HSI -> OPBS-64 + standardization -> MSNet residual E -> R = minmax(E)
    -> PCA-5 local spectral-spatial metric -> Magnitude prior M
    -> S = minmax(R * (1 + alpha * M))
```

Ground truth is not accepted by the detector API. If present in a `.mat` file,
it is loaded only after detection for metrics and figures.

## Repository layout

```text
MagCal/
|-- run.py                         # end-to-end MSNet + MagCal CLI
|-- magcal/                        # reusable Python package
|-- scripts/same_kernel_control.py # matched-kernel rarity control
|-- configs/paper_default.json     # frozen paper configuration
|-- docs/DATA.md                   # input format and data policy
|-- tests/test_smoke.py            # deterministic smoke tests
|-- requirements.txt
|-- environment.yml
|-- pyproject.toml
`-- LICENSE
```

## Installation

Python 3.10 is recommended.

```bash
git clone https://github.com/AR321109/MagCal.git
cd MagCal
conda env create -f environment.yml
conda activate magcal
pip install -e .
```

Alternatively:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

PyTorch/CUDA selection is environment-specific. Different PyTorch or CUDA
versions can produce small stochastic differences, so comparisons should use
the same environment and seed protocol.

## Data

The repository does not redistribute datasets. See [docs/DATA.md](docs/DATA.md).
The default loader expects a MATLAB file containing:

- `data`: `H x W x B` hyperspectral cube;
- `map`: optional `H x W` binary anomaly mask for evaluation only.

## Reproduce one MagCal run

Paper-budget command:

```bash
python run.py \
  --input data/abu-airport-1.mat \
  --output outputs/airport-1/seed-20261020 \
  --seed 20261020
```

CPU smoke command:

```bash
python run.py \
  --input data/example.mat \
  --output outputs/smoke \
  --bands 8 \
  --pca-components 3 \
  --core 4 \
  --epochs 1 \
  --iterations 1 \
  --output-iteration 1 \
  --device cpu
```

The frozen paper parameters are in `configs/paper_default.json`: OPBS-64,
PCA-5, `core=6`, `halo=1`, `ridge=1e-3`, `alpha=1`, five outer updates,
150 epochs per update, and seeds 20261020--20261024.

## Outputs

`run.py` writes:

- `maps.npz`: original score, Magnitude prior, calibrated score, optional GT;
- `metrics.json`: AP, ROC-AUC, FPR@95 and precision at top 1%;
- `run.json`: configuration, selected bands, losses and timing;
- `comparison.png`: HSI, GT, host score, prior and calibrated score;
- `candidate_changes.png`: top-budget anomaly/background entries and exits.

FPR@95 is the FPR at the first empirical ROC threshold whose TPR is at least
0.95; no interpolation is used.

## Matched-kernel control

The paper control keeps PCA, distances, median scaling, similarity kernel,
patches, scene normalization, host map and calibration equation fixed. It
replaces only the Magnitude solve with
`1 - mean(off-diagonal kernel similarity)`.

```bash
python scripts/same_kernel_control.py \
  --input data/abu-airport-1.mat \
  --maps outputs/airport-1/seed-20261020/maps.npz \
  --output outputs/airport-1/seed-20261020/rarity
```

On the seven reported ABU scenes, scene-macro AP is 0.33245 for original
MSNet, 0.34667 for matched-kernel rarity and 0.35340 for MagCal. MagCal minus
rarity is +0.00673 AP with a scene-paired 95% CI of [-0.00904, 0.02251], so
the additional advantage of the Magnitude solve is not established.

## Python API

```python
from magcal import MagCalConfig, MagCalDetector

config = MagCalConfig(seed=20261020)
detector = MagCalDetector(config).fit(cube)  # cube only; no GT argument
maps = detector.maps()
```

## Tests

```bash
python -m unittest discover -s tests -v
```

## Evidence boundary

The primary result is improved AP on seven evaluated ABU scenes. Mean FPR@95
also decreases for MagCal versus MSNet, but leave-one-scene-out checks show
that this effect is less stable than AP. Beach-4 is the reported failure case.
MagCal is an output-side calibrator, not an independent anomaly detector, and
the current evidence does not establish a universal advantage over every
same-kernel statistic, detector, dataset or operating point.

## License

Released under the MIT License. Dataset licenses remain with their providers.

## Citation and anonymity

Add the final paper citation after acceptance. For double-blind review, use an
anonymous repository mirror and verify that account names, commit metadata,
README text and archive metadata do not reveal author identity.
