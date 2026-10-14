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
Per-image MSE, PSNR and SSIM are written to `results/practice_dataset.csv`;
the terminal also shows averages by scenario and method. The dataset README
describes these as synthetic remote-sensing-style practice images, not real
Sentinel-2 data.

The EuroSAT training workflow is independent. Its models should be trained
before running `benchmark.py`; `evaluate_dataset.py` does not train or load
models.
