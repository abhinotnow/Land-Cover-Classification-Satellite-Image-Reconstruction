import numpy as np
import cv2


def _u8(x):
    return np.clip(x * 255 + 0.5, 0, 255).astype(np.uint8)


def _f(x):
    return x.astype(np.float32) / 255


# ---- denoisers ----
def gaussian(img, k=5, sigma=1.0):
    return cv2.GaussianBlur(img, (k, k), sigma)


def median(img, k=3):
    return _f(cv2.medianBlur(_u8(img), k))


def bilateral(img, d=5, sigma_color=0.1, sigma_space=3):
    return cv2.bilateralFilter(img.astype(np.float32), d, sigma_color, sigma_space)


DENOISERS = {"gaussian": gaussian, "median": median, "bilateral": bilateral}


# ---- inpainters (mask: 1 = missing) ----
def telea(img, mask, radius=3):
    return _f(cv2.inpaint(_u8(img), (mask > 0).astype(np.uint8), radius, cv2.INPAINT_TELEA))


def navier_stokes(img, mask, radius=3):
    return _f(cv2.inpaint(_u8(img), (mask > 0).astype(np.uint8), radius, cv2.INPAINT_NS))


INPAINTERS = {"telea": telea, "navier_stokes": navier_stokes}
