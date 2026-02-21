import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset.json")
MODEL_PATH = os.path.join(BASE_DIR, "luma_model.pth")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

# ===============================
# 1️⃣ Load Labeled Dataset
# ===============================

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)

X = []
y = []

for entry in dataset:
    X.append(entry["session_vector"])
    y.append(entry["label"])   # USE STORED LABEL

X = np.array(X)
y = np.array(y)

# Clip typing count (last feature)
X[:, -1] = np.clip(X[:, -1], 0, 300)

print("Total samples:", len(X))
print("Human samples:", sum(y == 1))
print("Bot samples:", sum(y == 0))

if len(X) < 2:
    raise SystemExit("Need at least 2 samples to train. Collect more data.")
if len(np.unique(y)) < 2:
    raise SystemExit("Need both human (1) and bot (0) labels. Add bot data via /generate_bot_data or collect both.")

# ===============================
# 2️⃣ Normalize Features
# ===============================

scaler = StandardScaler()
X = scaler.fit_transform(X)

# ===============================
# 3️⃣ Train/Test Split
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# ===============================
# 4️⃣ Model Definition
# ===============================

class LumaFusionModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

model = LumaFusionModel(X.shape[1])

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ===============================
# 5️⃣ Training
# ===============================

for epoch in range(60):
    model.train()
    output = model(X_train)
    loss = criterion(output, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

# ===============================
# 6️⃣ Evaluation
# ===============================

model.eval()
with torch.no_grad():
    predictions = model(X_test)
    probs = predictions.numpy()
    preds = (probs > 0.5).astype(int)

y_true = y_test.numpy()

acc = accuracy_score(y_true, preds)
precision = precision_score(y_true, preds, zero_division=0)
recall = recall_score(y_true, preds, zero_division=0)
f1 = f1_score(y_true, preds, zero_division=0)
auc = roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else 0.0
cm = confusion_matrix(y_true, preds)

print("\n===== MODEL EVALUATION =====")
print("Accuracy:", acc)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("AUC:", auc)
print("\nConfusion Matrix:")
print(cm)

# ===============================
# 7️⃣ Save Model + Scaler
# ===============================

torch.save(model.state_dict(), MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print("\nTraining complete.")
print("Model saved as luma_model.pth")
print("Scaler saved as scaler.pkl")
