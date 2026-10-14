"""Full performance matrix: Traditional vs ML/DL across PSNR, SSIM, RMSE and classifier acc/F1.
python benchmark.py --n 500"""
import argparse
import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

from lcr.data import load_eurosat, split, stratified_subset
from lcr.degrade import NOISES, cloud, scratches, estimate_mask
from lcr.classical import DENOISERS, INPAINTERS
from lcr.metrics import all_metrics
from lcr.ssrf import SSRF
from lcr.pconv import PConvUNet, restore
from lcr import classifier as C

if torch.cuda.is_available():
    DEV = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEV = "mps"
else:
    DEV = "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--n-validation", type=int, default=1000,
                    help="held-out images reserved for classifier validation")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs("results", exist_ok=True)

    X, y, _ = load_eurosat()
    _, te = split(y)
    val = stratified_subset(te, y, a.n_validation, seed=2)
    candidates = np.setdiff1d(te, val, assume_unique=False)
    te = stratified_subset(candidates, y, a.n, seed=a.seed)
    imgs = X[te].astype(np.float32) / 255
    ys = y[te]
    rng = np.random.default_rng(a.seed)

    clf = C.load("checkpoints/clf.pt", DEV)
    ssrf = SSRF.load("checkpoints/ssrf.joblib")
    pconv = PConvUNet().to(DEV)
    pconv.load_state_dict(torch.load("checkpoints/pconv.pt", map_location=DEV))
    pconv.eval()

    rows = []
    per_image_rows = []

    clean_pred = C.predict(clf, imgs, DEV)
    rows.append(dict(scenario="clean", method="none", n_images=len(imgs),
                     acc=float(accuracy_score(ys, clean_pred)),
                     f1=float(f1_score(ys, clean_pred, average="macro"))))
    for image_id, true_label, pred_label in zip(te, ys, clean_pred):
        per_image_rows.append({"image_id": int(image_id), "scenario": "clean",
                               "method": "none", "y_true": int(true_label),
                               "y_pred": int(pred_label),
                               "correct": bool(true_label == pred_label)})

    def log(scenario, method, restored):
        ms = [all_metrics(r, x) for r, x in zip(imgs, restored)]
        pred = C.predict(clf, restored, DEV)
        row = dict(scenario=scenario, method=method, n_images=len(restored),
                   **{k: float(np.mean([m[k] for m in ms])) for k in ms[0]},
                   **{f"{k}_std": float(np.std([m[k] for m in ms], ddof=1))
                      for k in ms[0]},
                   acc=float(accuracy_score(ys, pred)),
                   f1=float(f1_score(ys, pred, average="macro")))
        rows.append(row)
        for image_id, true_label, pred_label, metric in zip(te, ys, pred, ms):
            per_image_rows.append({"image_id": int(image_id), "scenario": scenario,
                                   "method": method, "y_true": int(true_label),
                                   "y_pred": int(pred_label),
                                   "correct": bool(true_label == pred_label), **metric})
        print(f"{scenario:11s} {method:14s} PSNR {row['psnr']:.2f}  SSIM {row['ssim']:.3f}  "
              f"RMSE {row['rmse']:.4f}  acc {row['acc']:.3f}  F1 {row['f1']:.3f}")

    # ---- Stage 1: denoising (classical) ----
    for name, nf in NOISES.items():
        deg = np.stack([nf(im, rng) for im in imgs])
        log(name, "degraded", deg)
        for dn, df in DENOISERS.items():
            log(name, dn, np.stack([df(d) for d in deg]))
        if name == "saltpepper":   # hybrid: classical outlier detector -> PConv U-Net
            sm = np.stack([estimate_mask(d) for d in deg])
            log(name, "pconv_seed", restore(pconv, deg, sm, DEV))

    # ---- Stage 2: inpainting ----
    holes = {"cloud": lambda im: cloud(im, rng, rng.uniform(0.1, 0.35)),
             "scratch": lambda im: scratches(im, rng)}
    for name, fn in holes.items():
        pairs = [fn(im) for im in imgs]
        deg = np.stack([p[0] for p in pairs])
        mask = np.stack([p[1] for p in pairs])
        smask = np.stack([estimate_mask(d) for d in deg])
        log(name, "degraded", deg)
        for n, f in INPAINTERS.items():
            log(name, n, np.stack([f(d, m) for d, m in zip(deg, mask)]))
        log(name, "ssrf", np.stack([ssrf.predict(d, m) for d, m in zip(deg, mask)]))
        log(name, "pconv_oracle", restore(pconv, deg, mask, DEV))
        log(name, "telea_seed", np.stack([INPAINTERS["telea"](d, m) for d, m in zip(deg, smask)]))
        log(name, "pconv_seed", restore(pconv, deg, smask, DEV))

    df = pd.DataFrame(rows)
    df.to_csv("results/benchmark.csv", index=False)
    per_image = pd.DataFrame(per_image_rows)
    per_image.to_csv("results/benchmark_per_image.csv", index=False)
    print("\n", df.round(4).to_string(index=False))
    print("\nSaved summary to results/benchmark.csv")
    print("Saved per-image records to results/benchmark_per_image.csv")


if __name__ == "__main__":
    main()
