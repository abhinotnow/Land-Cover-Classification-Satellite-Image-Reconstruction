"""Evaluate the supplied paired 256x256 practice dataset without training.

Usage: python evaluate_dataset.py [--data data/satellite_denoising_50_images]
Writes per-image results and a mean/std summary under results/.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from lcr.metrics import all_metrics


def read_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def luminance_transform(image, transform):
    """Apply a contrast transform to luminance while preserving RGB color."""
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = transform(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def methods(scenario, image):
    if scenario == "gaussian":
        return {
            "gaussian": cv2.GaussianBlur(image, (5, 5), 1.0),
            "median": cv2.medianBlur(image, 3),
            "bilateral": cv2.bilateralFilter(image, 5, 25, 3),
            "nlm": cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21),
        }
    if scenario == "salt_pepper":
        return {
            "gaussian": cv2.GaussianBlur(image, (5, 5), 1.0),
            "median": cv2.medianBlur(image, 3),
            "bilateral": cv2.bilateralFilter(image, 5, 25, 3),
            "nlm": cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21),
        }
    if scenario == "speckle":
        return {
            "gaussian": cv2.GaussianBlur(image, (5, 5), 1.0),
            "median": cv2.medianBlur(image, 3),
            "bilateral": cv2.bilateralFilter(image, 5, 25, 3),
            "nlm": cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21),
        }
    if scenario == "low_contrast":
        return {
            "histogram_equalization": luminance_transform(
                image, lambda y: cv2.equalizeHist(y)),
            "clahe": luminance_transform(
                image, lambda y: cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(y)),
        }
    raise ValueError(f"Unsupported scenario: {scenario}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/satellite_denoising_50_images"))
    parser.add_argument("--output", type=Path, default=Path("results/practice_dataset.csv"))
    parser.add_argument("--summary-output", type=Path,
                        default=Path("results/practice_dataset_summary.csv"))
    args = parser.parse_args()

    original_dir = args.data / "original"
    originals = {p.stem: p for p in original_dir.glob("*.png")}
    if not originals:
        raise FileNotFoundError(f"No PNG originals found in {original_dir}")

    rows = []
    scenarios = ("gaussian", "salt_pepper", "speckle", "low_contrast")
    for scenario in scenarios:
        scenario_dir = args.data / scenario
        for image_id, degraded_path in sorted(
                ((p.name.split("_")[0] + "_" + p.name.split("_")[1], p)
                 for p in scenario_dir.glob("*.png"))):
            if image_id not in originals:
                raise FileNotFoundError(f"No matching original for {degraded_path}")
            reference = read_rgb(originals[image_id]).astype(np.float32) / 255.0
            degraded_u8 = read_rgb(degraded_path)
            degraded = degraded_u8.astype(np.float32) / 255.0
            candidates = {"degraded": degraded_u8, **methods(scenario, degraded_u8)}
            for method, candidate_u8 in candidates.items():
                candidate = candidate_u8.astype(np.float32) / 255.0
                metrics = all_metrics(reference, candidate)
                rows.append({"image": image_id, "scenario": scenario,
                             "method": method, **metrics})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output, index=False)
    metrics = ["mse", "psnr", "ssim", "rmse"]
    summary = frame.groupby(["scenario", "method"])[metrics].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.insert(0, "n_images", frame.groupby(["scenario", "method"]).size())
    summary = summary.reset_index()
    summary.to_csv(args.summary_output, index=False)
    print(summary.round(4).to_string(index=False))
    print(f"\nSaved per-image results to {args.output}")
    print(f"Saved mean/std summary to {args.summary_output}")


if __name__ == "__main__":
    main()
