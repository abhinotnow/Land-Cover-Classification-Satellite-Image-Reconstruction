"""Classical-only demo on 3 images at 256x256.
Usage: python demo_classical.py [img1.png img2.png img3.png]
With no args it uses 3 EuroSAT test patches upscaled to 256 (blurry; use real
Sentinel-2 crops for the final report)."""
import sys
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lcr.data import load_eurosat, split
from lcr.degrade import NOISES, cloud, scratches
from lcr.classical import DENOISERS, INPAINTERS
from lcr.metrics import all_metrics


def get_images(paths):
    if paths:
        out = []
        for p in paths[:3]:
            im = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
            out.append(cv2.resize(im, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32) / 255)
        return out
    X, y, _ = load_eurosat()
    _, te = split(y)
    seen, out = set(), []
    for i in te:
        if y[i] not in seen and len(out) < 3:
            seen.add(y[i])
            out.append(cv2.resize(X[i], (256, 256), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255)
    return out


def main():
    os.makedirs("results", exist_ok=True)
    rng = np.random.default_rng(0)
    imgs = get_images(sys.argv[1:])
    rows = []

    # ---- denoising table ----
    for k, im in enumerate(imgs):
        for nname, nf in NOISES.items():
            deg = nf(im, rng)
            rows.append(dict(image=k, scenario=nname, method="degraded", **all_metrics(im, deg)))
            for dname, df in DENOISERS.items():
                rows.append(dict(image=k, scenario=nname, method=dname, **all_metrics(im, df(deg))))

    # ---- inpainting figure: Original -> Degraded -> Telea -> Navier-Stokes ----
    scenarios = (("cloud", lambda x: cloud(x, rng, 0.25)),
                 ("scratch", lambda x: scratches(x, rng, thick=(2, 4))))
    fig, axes = plt.subplots(len(imgs) * 2, 4, figsize=(12, 6 * len(imgs)))
    r = 0
    for k, im in enumerate(imgs):
        for sname, fn in scenarios:
            deg, mask = fn(im)
            panels = [("Original", im), ("Degraded", deg)] + \
                     [(n, f(deg, mask)) for n, f in INPAINTERS.items()]
            for c, (title, a) in enumerate(panels):
                axes[r, c].imshow(np.clip(a, 0, 1))
                axes[r, c].axis("off")
                if c == 0:
                    axes[r, c].set_title(f"img{k} / {sname}\nOriginal", fontsize=8)
                else:
                    m = all_metrics(im, a)
                    axes[r, c].set_title(f"{title}\nPSNR {m['psnr']:.2f}  SSIM {m['ssim']:.3f}", fontsize=8)
                    rows.append(dict(image=k, scenario=sname, method=title.lower().replace(" ", "_"), **m))
            r += 1
    plt.tight_layout()
    plt.savefig("results/demo_inpaint.png", dpi=150)

    df = pd.DataFrame(rows)
    df.to_csv("results/demo_classical.csv", index=False)
    print(df.groupby(["scenario", "method"])[["psnr", "ssim", "rmse"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
