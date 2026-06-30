import os
import joblib
import numpy as np
from features import extract_features
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

X_train = []
y_train = []

X_test = []
y_test = []

dataset_path = "dataset"

# Load training data
for folder, label in [("real", 0), ("spoof", 1)]:
    train_folder = os.path.join(dataset_path, "train", folder)

    for file in os.listdir(train_folder):
        image_path = os.path.join(train_folder, file)

        try:
            features = extract_features(image_path)
            X_train.append(features)
            y_train.append(label)
        except:
            print("Skipping train:", image_path)

# Load test data
for folder, label in [("real", 0), ("spoof", 1)]:
    test_folder = os.path.join(dataset_path, "test", folder)

    for file in os.listdir(test_folder):
        image_path = os.path.join(test_folder, file)

        try:
            features = extract_features(image_path)
            X_test.append(features)
            y_test.append(label)
        except:
            print("Skipping test:", image_path)

X_train = np.array(X_train)
y_train = np.array(y_train)

X_test = np.array(X_test)
y_test = np.array(y_test)

model = XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)
model.fit(X_train, y_train)

preds = model.predict(X_test)

accuracy = accuracy_score(y_test, preds)

print("Accuracy:", accuracy)

joblib.dump(model, "model.pkl")