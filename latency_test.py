import time
import joblib
from features import extract_features

model = joblib.load("model.pkl")

image_path = "dataset/test/real/20210327_15_09_21_000_7YC46d7RQSbz0KksQbvhM8Mbzlb2_T_4000_3000.jpg"

start = time.time()

features = extract_features(image_path).reshape(1, -1)
score = model.predict_proba(features)[0][1]

end = time.time()

latency_ms = (end - start) * 1000

print("Latency:", round(latency_ms, 2), "ms")