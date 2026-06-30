Spot the Fake Photo — Anti-Spoofing Image Classifier (Salescode Assignment)

This project implements a lightweight anti-spoofing image classifier to distinguish genuine photographs from recaptured images (photos of screens). Instead of using computationally expensive deep learning models, the solution uses classical computer vision feature engineering combined with XGBoost for fast, interpretable, and mobile-friendly inference.

Dataset 

The dataset consists of real and spoof image samples, split into training and testing sets. Each input image is resized to 224×224 before feature extraction.

Pipeline:

Image → Feature Extraction → XGBoost → Spoof Probability Score

Extracted features:

•	Laplacian Variance for sharpness detection
•	Sobel + Canny Edges for edge density analysis
•	Brightness Mean/Std for screen glow patterns
•	Glare Ratio for saturated highlight detection
•	FFT Energy for moiré/pixel-grid artefacts
•	LBP Histogram for micro-texture analysis
•	HOG Features for structural gradients

XGBoost was selected because it handles heterogeneous features efficiently, provides feature importance, and keeps the model lightweight for deployment.

Performance

Test Accuracy: 86.36%
Threshold: 0.70
Inference Latency: ~15–30 ms per image on a laptop CPU (Intel i5 / Ryzen 5 class), including feature extraction + XGBoost inference. This is fast enough to feel near-instant for end users.
Model Size: 634.39 KB

Cost Per Image

On-device deployment: ~$0.00 per image, since inference runs locally on the user’s phone/desktop without server dependency.
Cloud deployment: approximately $0.10–$0.25 per 1 million images, assuming:

CPU-only inference

~20 ms per image
Batch processing on a low-cost cloud instance (e.g., Amazon Web Services or Google Cloud)

The score distributions showed strong separation:

•	Real images mostly scored 0.00–0.10
•	Spoof images mostly scored 0.70–0.99
This threshold prioritizes spoof precision while maintaining strong recall.

Improvements with time:

1.	Upgrade to a lightweight CNN (like MobileNetV3) and ensemble it with XGBoost to improve accuracy beyond 90–95%. 
2.	Expand the dataset with varied lighting, angles, and devices to improve generalization.
3.	Introduce temporal signals (short video-based liveness detection) to make spoofing harder.

