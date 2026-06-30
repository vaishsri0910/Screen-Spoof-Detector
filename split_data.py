import os
import shutil
import random

source = "moire_classification"
target = "dataset"

classes = ["real", "spoof"]

for cls in classes:
    files = os.listdir(os.path.join(source, cls))
    random.shuffle(files)

    split_idx = int(0.8 * len(files))

    train_files = files[:split_idx]
    test_files = files[split_idx:]

    os.makedirs(os.path.join(target, "train", cls), exist_ok=True)
    os.makedirs(os.path.join(target, "test", cls), exist_ok=True)

    for file in train_files:
        shutil.copy(
            os.path.join(source, cls, file),
            os.path.join(target, "train", cls, file)
        )

    for file in test_files:
        shutil.copy(
            os.path.join(source, cls, file),
            os.path.join(target, "test", cls, file)
        )

print("Done splitting dataset.")