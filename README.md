# Satellite image processing

The repository includes two separate workflows:

- The EuroSAT RGB training and benchmark workflow in `train.py` and `benchmark.py`.
- The supplied 10-image practice dataset in `data/satellite_denoising_50_images`.

To evaluate the supplied paired images without training, run:

```bash
python evaluate_dataset.py
```

This compares the degraded inputs and classical filters (Gaussian, median,
bilateral and non-local means) for Gaussian, salt-and-pepper and speckle noise.
For low contrast images it compares luminance histogram equalization and CLAHE.
Per-image MSE, PSNR, SSIM and RMSE are written to
`results/practice_dataset.csv`; mean and sample standard deviation across the
10 images per scenario/method are saved to
`results/practice_dataset_summary.csv`. The dataset README
describes these as synthetic remote-sensing-style practice images, not real
Sentinel-2 data.

The EuroSAT training workflow is independent. Its models should be trained
before running `benchmark.py`; `evaluate_dataset.py` does not train or load
models.

## Quick EuroSAT model demonstration

For a short CPU demonstration, training defaults use a stratified subset of
3,000 EuroSAT training patches. The classifier validates on 1,000 balanced
held-out patches; the benchmark defaults to 100 balanced held-out patches.
These are demonstration metrics, not full-dataset results. SSRF was trained
separately and is saved at `checkpoints/ssrf.joblib`.

```bash
TORCH_HOME=/private/tmp/satellite-torch-cache .venv/bin/python train.py clf --epochs 3
.venv/bin/python train.py pconv --epochs 3
.venv/bin/python benchmark.py --n 100
```

The benchmark needs the classifier, SSRF, and PConv checkpoints. It writes
mean image-quality metrics plus sample standard deviations and aggregate
classification scores to `results/benchmark.csv`; each image-level record is
in `results/benchmark_per_image.csv`. Classifier validation images are excluded
from the benchmark subset. If MPS or CUDA is available, the training and
benchmark scripts select it automatically; otherwise they use CPU.
