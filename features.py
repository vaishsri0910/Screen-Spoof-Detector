import cv2
import numpy as np
from skimage.feature import local_binary_pattern, hog

def extract_features(image_path):
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    img = cv2.resize(img, (256, 256))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    features = []

    # Laplacian variance
    features.append(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Sobel gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

    features.append(np.mean(np.abs(sobelx)))
    features.append(np.mean(np.abs(sobely)))
    features.append(np.std(np.abs(sobelx)))
    features.append(np.std(np.abs(sobely)))

    # Edge density
    edges = cv2.Canny(gray, 100, 200)
    features.append(np.sum(edges > 0) / edges.size)

    # Brightness stats
    features.append(np.mean(gray))
    features.append(np.std(gray))

    # Glare ratio
    features.append(np.sum(gray > 240) / gray.size)

    # FFT features
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.log(np.abs(fft_shift) + 1)

    features.append(np.mean(magnitude))
    features.append(np.std(magnitude))
    features.append(np.max(magnitude))

    # LBP histogram
    lbp = local_binary_pattern(gray, 16, 2, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=18, range=(0, 18))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)

    features.extend(lbp_hist)

    # HOG features
    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        feature_vector=True
    )

    # Use only summary stats (keeps model small)
    features.append(np.mean(hog_features))
    features.append(np.std(hog_features))
    features.append(np.max(hog_features))

    return np.array(features)