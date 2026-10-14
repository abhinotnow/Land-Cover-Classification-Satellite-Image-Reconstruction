"""python train.py clf --epochs 5 | ssrf | pconv --epochs 20"""
import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from lcr.data import load_eurosat, split
from lcr import classifier as C
from lcr.degrade import random_mask
from lcr.pconv import PConvUNet
from lcr.ssrf import SSRF

DEV = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("checkpoints", exist_ok=True)


def train_clf(X, y, tr, te, epochs):
    model = C.build().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), 3e-4)
    Xte = X[te].astype(np.float32) / 255
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(tr)
        tot = 0.0
        for i in range(0, len(perm), 128):
            b = perm[i:i + 128]
            x = X[b].astype(np.float32) / 255
            if np.random.rand() < 0.5:
                x = x[:, :, ::-1]
            if np.random.rand() < 0.5:
                x = x[:, ::-1]
            loss = F.cross_entropy(model(C.to_tensor(x, DEV)), torch.from_numpy(y[b]).to(DEV))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        s = C.score(model, Xte, y[te], DEV)
        print(f"clf ep{ep + 1}: loss {tot / len(perm):.4f}  test acc {s['acc']:.4f}  f1 {s['f1']:.4f}")
    torch.save(model.state_dict(), "checkpoints/clf.pt")


def train_ssrf(X, tr, n_imgs=2000):
    imgs = X[tr[:n_imgs]].astype(np.float32) / 255
    SSRF().fit(imgs).save("checkpoints/ssrf.joblib")
    print("saved checkpoints/ssrf.joblib")


class HoleSet(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        rng = np.random.default_rng()
        img = self.X[i].astype(np.float32) / 255
        if rng.random() < 0.5:
            img = img[:, ::-1]
        m = random_mask(img.shape[:2], rng)                  # 1 = missing
        gt = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)
        valid = torch.from_numpy(1 - m)[None]                # 1 = valid
        return gt, valid


def pconv_loss(out, gt, valid):
    hole = 1 - valid
    comp = out * hole + gt * valid
    tv = (comp[:, :, :, 1:] - comp[:, :, :, :-1]).abs().mean() + \
         (comp[:, :, 1:, :] - comp[:, :, :-1, :]).abs().mean()
    return 6 * F.l1_loss(out * hole, gt * hole) + F.l1_loss(out * valid, gt * valid) + 0.1 * tv


def train_pconv(X, tr, epochs):
    model = PConvUNet().to(DEV)
    opt = torch.optim.Adam(model.parameters(), 2e-4)
    dl = DataLoader(HoleSet(X[tr]), batch_size=64, shuffle=True, num_workers=2, drop_last=True)
    for ep in range(epochs):
        model.train()
        tot, n = 0.0, 0
        for gt, valid in dl:
            gt, valid = gt.to(DEV), valid.to(DEV)
            loss = pconv_loss(model(gt * valid, valid), gt, valid)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            n += 1
        print(f"pconv ep{ep + 1}: loss {tot / n:.4f}")
        torch.save(model.state_dict(), "checkpoints/pconv.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["clf", "ssrf", "pconv"])
    ap.add_argument("--epochs", type=int, default=5)
    a = ap.parse_args()
    X, y, _ = load_eurosat()
    tr, te = split(y)
    if a.what == "clf":
        train_clf(X, y, tr, te, a.epochs)
    elif a.what == "ssrf":
        train_ssrf(X, tr)
    else:
        train_pconv(X, tr, a.epochs)
