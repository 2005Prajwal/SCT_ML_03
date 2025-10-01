# train_fast.py
import os
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import joblib
from dataset import load_dataset  # your dataset.py

# -----------------------------
# Load dataset
# -----------------------------
print("📥 Loading dataset...")
X, y = load_dataset("dataset/train")

X = np.array(X)
y = np.array(y)

# Shuffle dataset
X, y = shuffle(X, y, random_state=42)

# -----------------------------
# Reduce dataset for fast training
# -----------------------------
cats_idx = np.where(y == 0)[0]
dogs_idx = np.where(y == 1)[0]
min_count = min(len(cats_idx), len(dogs_idx), 500)  # 500 per class for speed

selected_idx = np.concatenate((cats_idx[:min_count], dogs_idx[:min_count]))
X = X[selected_idx]
y = y[selected_idx]

# Shuffle again
X, y = shuffle(X, y, random_state=42)

print(f"✅ Using {len(X)} images for fast training")
print("Class distribution:", np.bincount(y))  # [num_cats, num_dogs]

# -----------------------------
# Flatten images
# -----------------------------
X = X.reshape(len(X), -1)

# -----------------------------
# Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# -----------------------------
# Train LinearSVC
# -----------------------------
print("⚡ Training LinearSVC ")
model = LinearSVC(max_iter=5000)  # increase max_iter to ensure convergence
model.fit(X_train, y_train)

# -----------------------------
# Evaluate
# -----------------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"🎯 Accuracy on test set: {acc:.4f}")

# -----------------------------
# Save model
# -----------------------------
joblib.dump(model, "cat_dog_svm_fast.pkl")
print("💾  model saved as cat_dog_svm_fast.pkl")
