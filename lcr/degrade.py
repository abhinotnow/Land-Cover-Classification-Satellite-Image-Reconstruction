"""Conventions: images are float32 HWC RGB in [0,1]. Masks are float32 HW, 1 = MISSING."""
import numpy as np
import cv2

# ---------- noise ----------
def gaussian_noise(img, rng, sigma=0.08):
    return np.clip(img + rng.normal(0, sigma, img.shape), 0, 1).astype(np.float32)

def salt_pepper(img, rng, p=0.05):
    out = img.copy()
    r = rng.random(img.shape[:2])
    out[r < p / 2] = 0.0
    out[(r >= p / 2) & (r < p)] = 1.0
    return out

def speckle(img, rng, var=0.1):
    return np.clip(img + img * rng.normal(0, np.sqrt(var), img.shape), 0, 1).astype(np.float32)

NOISES = {"gaussian": gaussian_noise, "saltpepper": salt_pepper, "speckle": speckle}

# ---------- missing-data masks ----------
def cloud(img, rng, coverage=0.25):
    """Blobby synthetic cloud. Returns (degraded, mask)."""
    h, w = img.shape[:2]
    n = rng.random((max(h // 8, 4), max(w // 8, 4))).astype(np.float32)
    n = cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)
    n = cv2.GaussianBlur(n, (0, 0), sigmaX=h / 32)
    n = (n - n.min()) / (n.max() - n.min() + 1e-8)
    thr = np.quantile(n, 1 - coverage)
    alpha = np.clip((n - thr) * 10, 0, 1)[..., None]
    col = np.clip(0.92 + 0.02 * rng.standard_normal((h, w, 1)), 0, 1)
    deg = img * (1 - alpha) + col * alpha
    mask = (alpha[..., 0] > 0.5).astype(np.float32)
    return deg.astype(np.float32), mask

def scratches(img, rng, n=None, thick=(1, 2)):
    """Narrow line dropouts (dead sensor lines), set to 0. Returns (degraded, mask)."""
    h, w = img.shape[:2]
    m = np.zeros((h, w), np.uint8)
    for _ in range(n or int(rng.integers(3, 8))):
        p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        p2 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        cv2.line(m, p1, p2, 1, int(rng.integers(thick[0], thick[1] + 1)))
    deg = img.copy()
    deg[m > 0] = 0.0
    return deg, m.astype(np.float32)

def random_mask(shape, rng):
    """Training masks for the PConv U-Net: clouds, scratches, or both."""
    h, w = shape
    dummy = np.zeros((h, w, 3), np.float32)
    kind = rng.random()
    if kind < 0.4:
        return cloud(dummy, rng, rng.uniform(0.05, 0.4))[1]
    if kind < 0.7:
        return scratches(dummy, rng)[1]
    m = cloud(dummy, rng, rng.uniform(0.05, 0.3))[1]
    return np.maximum(m, scratches(dummy, rng)[1])

# ---------- classical "seed mask" (the hybrid step) ----------
def estimate_mask(img, bright=0.5, spread=0.2, outlier=0.3):
    """Fast classical detector of missing/corrupt pixels -> mask (1 = missing).
    - whiteness test  : bright & low colour spread  -> cloud
    - median outlier  : |img - median3x3| large     -> impulse noise / thin lines
    - dead pixels     : exactly zero in all bands   -> sensor dropout
    Thresholds are heuristics; tune on your data."""
    g = cv2.GaussianBlur(img, (3, 3), 0)
    white = (g.mean(2) > bright) & ((g.max(2) - g.min(2)) < spread)
    u8 = np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8)
    med = cv2.medianBlur(u8, 3).astype(np.float32) / 255
    imp = np.abs(img - med).max(2) > outlier
    dead = img.max(2) <= 0.0
    m = (white | imp | dead).astype(np.uint8)
    m = cv2.dilate(m, np.ones((3, 3), np.uint8))
    return m.astype(np.float32)
