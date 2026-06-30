import time
import joblib
import os
import random
from features import extract_features

model = joblib.load("model.pkl")

# pick random image from dataset
folders = [
    "dataset/test/real",
    "dataset/test/spoof"
]

all_images = []

for folder in folders:
    for f in os.listdir(folder):
        all_images.append(os.path.join(folder, f))

img = random.choice(all_images)

start = time.time()

features = extract_features(img).reshape(1, -1)
model.predict_proba(features)

end = time.time()

print("Image used:", img)
print("Latency:", round((end - start) * 1000, 2), "ms")