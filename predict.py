import cv2
import pickle
import sys
from feature_extractor import extract_features

THRESHOLD = 0.70

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

print("Model classes:", model.classes_)

# Read image
image_path = sys.argv[1]
img = cv2.imread(image_path)

if img is None:
    print("Invalid image path")
    exit()

features = extract_features(img)

# Predict probabilities
probs = model.predict_proba([features])[0]
prediction = model.predict([features])[0]

print("Probabilities:", probs)

# Dynamic label mapping
label = "REAL" if prediction == 1 else "SPOOF"

print(f"Prediction: {label}")
print(f"Confidence: {max(probs):.2f}")