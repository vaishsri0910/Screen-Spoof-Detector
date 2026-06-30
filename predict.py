import sys
import joblib
from features import extract_features

model = joblib.load("model.pkl")

image_path = sys.argv[1]

features = extract_features(image_path).reshape(1, -1)

prob = model.predict_proba(features)[0][1]

print(round(float(prob), 2))