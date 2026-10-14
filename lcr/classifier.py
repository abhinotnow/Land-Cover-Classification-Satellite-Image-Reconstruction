import numpy as np
import torch
import torch.nn as nn
import torchvision
from sklearn.metrics import f1_score, accuracy_score

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def build(num_classes=10, pretrained=True):
    w = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
    m = torchvision.models.resnet18(weights=w)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def to_tensor(x, device):
    """x: float32 (N,H,W,3) in [0,1]"""
    t = torch.from_numpy(np.ascontiguousarray(x)).permute(0, 3, 1, 2).float()
    return ((t - MEAN) / STD).to(device)


@torch.no_grad()
def predict(model, X, device, bs=256):
    model.eval()
    return np.concatenate([model(to_tensor(X[i:i + bs], device)).argmax(1).cpu().numpy()
                           for i in range(0, len(X), bs)])


def score(model, X, y, device):
    p = predict(model, X, device)
    return {"acc": accuracy_score(y, p), "f1": f1_score(y, p, average="macro")}


def load(path, device):
    m = build(pretrained=False)
    m.load_state_dict(torch.load(path, map_location=device))
    return m.to(device).eval()
