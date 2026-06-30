import sys
import joblib
import numpy as np
from features import extract_features

model = joblib.load("model.pkl")

image_path = sys.argv[1]

features = extract_features(image_path).reshape(1, -1)

score = model.predict_proba(features)[0][1]

print(round(float(score), 2))