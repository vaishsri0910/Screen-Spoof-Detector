# Screen Spoof Detector

A lightweight anti-spoofing image classifier that detects whether an image is a **real photograph** or a **screen recapture (spoof)** using classical computer vision and XGBoost.

Built as part of the **Salescode Assignment**.

---

## Overview

Screen spoofing is a common fraud technique where attackers present photos of screens instead of genuine images (for example in identity verification systems). This project aims to detect such spoof attempts efficiently.

Instead of deep learning, this solution uses handcrafted image features + XGBoost for:

* Faster inference
* Lower deployment cost
* Better interpretability
* Smaller model size

---

## Features

✔ Upload-based spoof detection
✔ Continuous live camera spoof detection
✔ Lightweight classical CV pipeline
✔ Fast real-time inference
✔ Mobile/edge deployment friendly
✔ Low cloud cost

---

## Pipeline

Image → Feature Extraction → XGBoost → Spoof Probability Score

### Extracted Features

This model uses 5 handcrafted features:

### 1. FFT Energy

Detects periodic screen artefacts and pixel-grid patterns.

### 2. Noise Estimation

Real photos usually contain more natural sensor noise.

### 3. Edge Ratio

Measures sharp UI-like screen edges using Canny edge detection.

### 4. Brightness Variance

Screens often produce unnatural brightness patterns.

### 5. Texture Smoothness

Uses Laplacian variance to detect unnatural smoothness.

---

## Model Details

* **Model:** XGBoost Classifier
* **Input Size:** 224 × 224
* **Output:** Binary classification
* **Threshold:** 0.70
* **Serialized Model Size:** **634.39 KB**

---

## Performance

| Metric                      | Value                    |
| --------------------------- | ------------------------ |
| Test Accuracy               | 86.36%                   |
| Inference Latency           | ~15–30 ms/image          |
| Model Size                  | 634.39 KB                |
| Deployment Cost (On-device) | ~$0.00                   |
| Deployment Cost (Cloud)     | ~$0.10–$0.25 / 1M images |

### Device Used

Laptop CPU (Intel/Ryzen class)

---

## Project Structure

```text
Screen-Spoof-Detector/
│── app.py
│── predict.py
│── feature_extractor.py
│── model.pkl
│── requirements.txt
│── README.md
```

---

## Requirements

Install all dependencies:

```bash
pip install -r requirements.txt
```

Required packages:

* opencv-python
* numpy
* scipy
* xgboost
* pillow
* streamlit
* streamlit-webrtc
* av

---

## Installation

Clone the repository:

```bash
git clone https://github.com/vaishsri0910/Screen-Spoof-Detector.git
cd Screen-Spoof-Detector
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

## 1. Run CLI Prediction

Use:

```bash
python predict.py image.jpg
```

Example output:

```text
Prediction: REAL
Confidence: 0.92
```

---

## 2. Run Streamlit Web App

Use:

```bash
streamlit run app.py
```

This opens a web app with two modes:

### Upload Image Mode

Upload any image to test spoof probability.

### Live Camera Mode

Runs continuous real-time spoof detection through webcam.

Shows:

```text
REAL: 0.91
SPOOF: 0.87
```

live on screen.

---

## How Live Detection Works

The webcam captures frames continuously.

For each frame:

1. Extract handcrafted features
2. Run XGBoost prediction
3. Display label + confidence in real time

This makes testing interactive and practical for evaluators.

---

## Future Improvements

Given more time, the following upgrades can improve performance:

* Hybrid CNN + XGBoost pipeline
* MobileNetV3 integration
* Larger and more diverse dataset
* Temporal liveness detection
* Confidence calibration
* Hard-negative mining
* Adversarial robustness training

---

## Business Impact

This system can help:

* Prevent KYC fraud
* Detect identity spoofing attacks
* Reduce verification costs
* Enable on-device anti-spoofing
* Improve fraud detection pipelines

---

## Why Classical CV?

A CNN would likely improve accuracy, but this pipeline was chosen because it:

* Requires no GPU
* Is highly interpretable
* Has low latency
* Uses very little memory
* Is easier to deploy on mobile

This makes it ideal for lightweight production use.

---

## Demo

Supports:

✔ Static image upload
✔ Continuous webcam testing
✔ Real-time confidence scoring

---

## Author

**Vaishnavi Srivastava**
Final Year CSE Student | Frontend Developer | GenAI Enthusiast
