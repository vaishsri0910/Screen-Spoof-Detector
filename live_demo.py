import cv2
import pickle
from feature_extractor import extract_features

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

cap = cv2.VideoCapture(0)

print("Press Q to quit")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not access webcam")
        break

    # Extract features
    features = extract_features(frame)

    # Predict spoof probability
    prob = model.predict_proba([features])[0][1]

    # Label
    label = "SPOOF" if prob > 0.70 else "REAL"

    # Put text on screen
    cv2.putText(
        frame,
        f"{label}: {prob:.2f}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Live Anti-Spoof Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()