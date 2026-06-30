import os
import numpy as np
import joblib
from features import extract_features
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

data_dir = "dataset/train"

X, y = [], []

for label, cls in enumerate(["real", "spoof"]):
    folder = os.path.join(data_dir, cls)

    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        try:
            X.append(extract_features(path))
            y.append(label)
        except:
            continue

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9
)

model.fit(X_train, y_train)

acc = model.score(X_test, y_test)
print("Accuracy:", acc)

joblib.dump(model, "model.pkl")