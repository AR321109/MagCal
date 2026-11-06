# Data preparation

The repository does not redistribute hyperspectral datasets. Prepare one MATLAB
file per scene with:

- `data`: an `H x W x B` finite numeric hyperspectral cube;
- `map`: an optional `H x W` binary anomaly mask used only for evaluation.

Example inspection:

```python
from scipy.io import loadmat

scene = loadmat("airport-1.mat")
print(scene["data"].shape, scene["map"].shape)
```

Use `--data-key` and `--gt-key` when a source dataset uses other variable names.
An empty `--gt-key ""` disables ground-truth loading. Dataset licenses and
download conditions remain those of the original providers.
