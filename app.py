import streamlit as st
import cv2
import pickle
import numpy as np
import av
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from feature_extractor import extract_features

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Spot the Fake Photo")
st.write("Anti-Spoofing Image Classifier")

st.write("Model Classes:", model.classes_)


def get_label(pred):
    """
    Assumption:
    1 = REAL
    0 = SPOOF
    If your training used opposite labels,
    just swap here.
    """
    return "REAL" if pred == 1 else "SPOOF"


def predict_frame(img):
    features = extract_features(img)

    probs = model.predict_proba([features])[0]
    pred = model.predict([features])[0]

    label = get_label(pred)
    confidence = max(probs)

    return label, confidence


# Upload mode
mode = st.radio(
    "Choose Mode",
    ["Upload Image", "Live Camera"]
)

if mode == "Upload Image":
    uploaded_file = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        img = Image.open(uploaded_file)
        img_np = np.array(img)

        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        st.image(img)

        label, confidence = predict_frame(img_cv)

        st.write(f"Prediction: {label}")
        st.write(f"Confidence: {confidence:.2f}")

        if label == "SPOOF":
            st.error("⚠ Spoof Detected")
        else:
            st.success("✅ Genuine Photo")


# Live Camera Mode
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        label, confidence = predict_frame(img)

        cv2.putText(
            img,
            f"{label}: {confidence:.2f}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


if mode == "Live Camera":
    webrtc_streamer(
        key="camera",
        video_processor_factory=VideoProcessor
    )