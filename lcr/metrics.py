import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def psnr(ref, x):
    return float(peak_signal_noise_ratio(ref, x, data_range=1.0))


def ssim(ref, x):
    return float(structural_similarity(ref, x, channel_axis=2, data_range=1.0))


def rmse(ref, x):
    return float(np.sqrt(np.mean((ref - x) ** 2)))


def mse(ref, x):
    return float(np.mean((ref - x) ** 2))


def all_metrics(ref, x):
    return {"mse": mse(ref, x), "psnr": psnr(ref, x),
            "ssim": ssim(ref, x), "rmse": rmse(ref, x)}
