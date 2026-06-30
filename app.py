import streamlit as st
import joblib
from features import extract_features
import tempfile
import os

# Page config
st.set_page_config(
    page_title="Spot the Fake Photo",
    page_icon="",
    layout="centered"
)

# Load model
model = joblib.load("model.pkl")

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
        color: white;
    }
    .title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        color: #A0A0A0;
        margin-bottom: 30px;
    }
    .result-box {
        padding: 20px;
        border-radius: 12px;
        margin-top: 20px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown(
    '<div class="title">Spot the Fake Photo</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Detect whether an image is a REAL photo or a PHOTO OF A SCREEN</div>',
    unsafe_allow_html=True
)

# Upload card
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(
            uploaded_file,
            caption="Uploaded Image",
            use_container_width=True
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    features = extract_features(temp_path).reshape(1, -1)

    score = model.predict_proba(features)[0][1]

    threshold = 0.70

    with col2:
        st.metric("Spoof Probability", f"{score:.2f}")

        if score >= threshold:
            st.error("⚠️ Photo of a Screen Detected")
            st.progress(min(float(score), 1.0))
        else:
            st.success("✅ Real Photo")
            st.progress(min(float(score), 1.0))

    # Confidence explanation
    st.markdown("### Analysis")
    st.write(f"""
    - **Score:** {score:.2f}  
    - **Threshold:** {threshold}  
    - Lower score → more likely real  
    - Higher score → more likely spoof  
    """)

    os.remove(temp_path)