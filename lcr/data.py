import os
import numpy as np
from sklearn.model_selection import train_test_split

CACHE = "data/eurosat_rgb.npz"

def load_eurosat(root="data"):
    """Returns X uint8 (N,64,64,3), y int (N,), class names."""
    if os.path.exists(CACHE):
        d = np.load(CACHE)
        return d["X"], d["y"], list(d["classes"])
    from torchvision.datasets import EuroSAT
    os.makedirs(root, exist_ok=True)
    ds = EuroSAT(root, download=True)
    X, y = [], []
    for im, lbl in ds:
        X.append(np.asarray(im.convert("RGB")))
        y.append(lbl)
    X, y = np.stack(X), np.array(y)
    np.savez_compressed(CACHE, X=X, y=y, classes=np.array(ds.classes))
    return X, y, list(ds.classes)

def split(y, seed=0, test=0.2):
    """Class-wise (stratified) 80/20 split, as in the EuroSAT paper."""
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=test, stratify=y, random_state=seed)
    return tr, te


def stratified_subset(indices, y, n, seed=0):
    """Sample up to n indices while preserving class proportions."""
    indices = np.asarray(indices)
    if n is None or n >= len(indices):
        return indices.copy()
    if n <= 0:
        raise ValueError("subset size must be positive")

    rng = np.random.default_rng(seed)
    labels = np.unique(y[indices])
    groups = [indices[y[indices] == label] for label in labels]
    exact = np.array([len(group) * n / len(indices) for group in groups])
    quotas = np.floor(exact).astype(int)
    remainder = n - int(quotas.sum())
    for j in np.argsort(-(exact - quotas))[:remainder]:
        quotas[j] += 1

    selected = [rng.choice(group, size=int(quota), replace=False)
                for group, quota in zip(groups, quotas) if quota]
    result = np.concatenate(selected)
    rng.shuffle(result)
    return result
