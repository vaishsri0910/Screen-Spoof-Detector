import cv2
import numpy as np
from scipy.fft import fft2


def extract_features(img):
    if img is None:
        raise ValueError("Invalid image")

    img = cv2.resize(img, (224, 224))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # FFT energy
    f = fft2(gray)
    fshift = np.abs(f)
    fft_energy = np.mean(fshift)

    # Noise estimation
    noise = np.std(gray - cv2.GaussianBlur(gray, (5, 5), 0))

    # Edge ratio
    edges = cv2.Canny(gray, 100, 200)
    edge_ratio = np.sum(edges) / 255 / (224 * 224)

    # Brightness variance
    brightness_var = np.var(gray)

    # Texture smoothness
    smoothness = cv2.Laplacian(gray, cv2.CV_64F).var()

    return np.array([
        fft_energy,
        noise,
        edge_ratio,
        brightness_var,
        smoothness
    ])