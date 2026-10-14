"""Spatio-Spectral Random Forest: predicts a missing pixel's RGB from its 3x3 neighbourhood.
Holes are filled outside-in (onion peeling), so large holes work iteratively."""
import numpy as np
import joblib
from scipy.ndimage import binary_dilation
from sklearn.ensemble import RandomForestRegressor

OFFS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _feats(img, known, ys, xs):
    ip = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode="edge")
    kp = np.pad(known, 1, mode="edge")
    nv = np.stack([ip[ys + 1 + dy, xs + 1 + dx] for dy, dx in OFFS], 1)   # N,8,3
    nk = np.stack([kp[ys + 1 + dy, xs + 1 + dx] for dy, dx in OFFS], 1)   # N,8
    cnt = nk.sum(1)
    mean = (nv * nk[..., None]).sum(1) / np.maximum(cnt, 1)[:, None]      # N,3
    nv = np.where(nk[..., None], nv, mean[:, None, :])                    # impute unknown nbrs
    X = np.concatenate([(nv - mean[:, None, :]).reshape(len(ys), -1),
                        nk.astype(np.float32), mean], 1)                  # 24 + 8 + 3
    return X, mean, cnt


class SSRF:
    def __init__(self, n_estimators=60, max_depth=16):
        self.rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                        min_samples_leaf=3, max_features=0.5,
                                        n_jobs=-1, random_state=0)

    def fit(self, imgs, per_img=150, seed=0):
        """imgs: float32 (N,H,W,3) clean images."""
        rng = np.random.default_rng(seed)
        _, H, W, _ = imgs.shape
        Xs, Ys = [], []
        for im in imgs:
            known = rng.random((H, W)) < rng.uniform(0.3, 0.9)
            ys = rng.integers(1, H - 1, per_img)   # interior only: no border leakage
            xs = rng.integers(1, W - 1, per_img)
            X, mean, cnt = _feats(im, known, ys, xs)
            ok = cnt > 0
            Xs.append(X[ok])
            Ys.append((im[ys, xs] - mean)[ok])
        self.rf.fit(np.concatenate(Xs), np.concatenate(Ys))
        return self

    def predict(self, img, mask):
        """img float32 HWC, mask 1=missing -> restored image."""
        known = mask == 0
        out = img.copy()
        while not known.all():
            front = binary_dilation(known, structure=np.ones((3, 3), bool)) & ~known
            if not front.any():
                break
            ys, xs = np.nonzero(front)
            X, mean, _ = _feats(out, known, ys, xs)
            out[ys, xs] = np.clip(mean + self.rf.predict(X).reshape(-1, 3), 0, 1)
            known[ys, xs] = True
        return out

    def save(self, path):
        joblib.dump(self.rf, path)

    @classmethod
    def load(cls, path):
        s = cls()
        s.rf = joblib.load(path)
        return s
