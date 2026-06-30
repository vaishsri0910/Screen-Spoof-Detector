import streamlit as st
import joblib
import tempfile
from PIL import Image
from features import extract_features

model = joblib.load("model.pkl")

st.title("📷 Spot the Fake Photo")

threshold = st.slider("Threshold", 0.0, 1.0, 0.7)

uploaded = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded:
    image = Image.open(uploaded)
    st.image(image, use_container_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded.getvalue())
        path = tmp.name

    features = extract_features(path).reshape(1, -1)
    score = model.predict_proba(features)[0][1]

    st.metric("Spoof Score", round(score, 2))

    if score > threshold:
        st.error("Photo of Screen Detected")
    else:
        st.success("Real Photo")