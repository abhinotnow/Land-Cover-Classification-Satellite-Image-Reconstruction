"""Full performance matrix: Traditional vs ML/DL across PSNR, SSIM, RMSE and classifier acc/F1.
python benchmark.py --n 500"""
import argparse
import os
import numpy as np
import pandas as pd
import torch

from lcr.data import load_eurosat, split
from lcr.degrade import NOISES, cloud, scratches, estimate_mask
from lcr.classical import DENOISERS, INPAINTERS
from lcr.metrics import all_metrics
from lcr.ssrf import SSRF
from lcr.pconv import PConvUNet, restore
from lcr import classifier as C

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs("results", exist_ok=True)

    X, y, _ = load_eurosat()
    _, te = split(y)
    te = te[:a.n]
    imgs = X[te].astype(np.float32) / 255
    ys = y[te]
    rng = np.random.default_rng(a.seed)

    clf = C.load("checkpoints/clf.pt", DEV)
    ssrf = SSRF.load("checkpoints/ssrf.joblib")
    pconv = PConvUNet().to(DEV)
    pconv.load_state_dict(torch.load("checkpoints/pconv.pt", map_location=DEV))
    pconv.eval()

    rows = [dict(scenario="clean", method="none", **C.score(clf, imgs, ys, DEV))]

    def log(scenario, method, restored):
        ms = [all_metrics(r, x) for r, x in zip(imgs, restored)]
        row = dict(scenario=scenario, method=method,
                   **{k: float(np.mean([m[k] for m in ms])) for k in ms[0]},
                   **C.score(clf, restored, ys, DEV))
        rows.append(row)
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
    print("\n", df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
