import copy
import json
import os
import random

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset.json")
MODEL_PATH = os.path.join(BASE_DIR, "luma_model.pth")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
CALIBRATOR_PATH = os.path.join(BASE_DIR, "calibrator.pkl")
THRESHOLD_PATH = os.path.join(BASE_DIR, "threshold.json")

SEED = 42
TEST_SIZE = 0.2
VAL_SIZE_FROM_TRAINVAL = 0.25  # 60/20/20 train/val/test
MAX_EPOCHS = 220
EARLY_STOP_PATIENCE = 26
EARLY_STOP_MIN_DELTA = 1e-4
LEARNING_RATE = 8e-4
WEIGHT_DECAY = 3e-4
THRESHOLD_GRID = np.round(np.arange(0.30, 0.76, 0.01), 2)
MIN_HUMAN_RECALL = 0.78


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LumaFusionModel(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def safe_auc(y_true: np.ndarray, probs: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.0
    return float(roc_auc_score(y_true, probs))


def threshold_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def choose_threshold(y_true: np.ndarray, probs: np.ndarray) -> tuple[dict, list[dict]]:
    sweep = [threshold_metrics(y_true, probs, t) for t in THRESHOLD_GRID]
    eligible = [m for m in sweep if m["recall"] >= MIN_HUMAN_RECALL]
    pool = eligible if eligible else sweep
    best = max(pool, key=lambda m: (m["f1"], m["recall"], m["accuracy"], m["precision"]))
    return best, sweep


def print_row(prefix: str, m: dict) -> None:
    cm = m["confusion_matrix"]
    print(
        f"{prefix} thr={m['threshold']:.2f} | "
        f"acc={m['accuracy']:.3f} | prec={m['precision']:.3f} | "
        f"rec={m['recall']:.3f} | f1={m['f1']:.3f} | "
        f"TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}"
    )


def infer_probs(model: nn.Module, x_tensor: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(x_tensor).squeeze(1).cpu().numpy()


set_seed(SEED)

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)

X = np.array([entry["session_vector"] for entry in dataset], dtype=np.float32)
y = np.array([entry["label"] for entry in dataset], dtype=np.int64)

# Clip typing count (last feature)
X[:, -1] = np.clip(X[:, -1], 0, 300)

print("Total samples:", len(X))
print("Human samples:", int(np.sum(y == 1)))
print("Bot samples:", int(np.sum(y == 0)))

if len(X) < 40:
    raise SystemExit("Need at least 40 samples to train a stable model.")
if len(np.unique(y)) < 2:
    raise SystemExit("Need both human (1) and bot (0) labels.")

X_train_val_raw, X_test_raw, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=SEED,
    stratify=y,
)

X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    X_train_val_raw,
    y_train_val,
    test_size=VAL_SIZE_FROM_TRAINVAL,
    random_state=SEED,
    stratify=y_train_val,
)

print("\n===== DATA SPLIT =====")
print("Train samples:", len(X_train_raw))
print("Validation samples:", len(X_val_raw))
print("Test samples:", len(X_test_raw))

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_val = scaler.transform(X_val_raw)
X_test = scaler.transform(X_test_raw)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
X_test_t = torch.tensor(X_test, dtype=torch.float32)

model = LumaFusionModel(X.shape[1])
bce = nn.BCELoss(reduction="none")
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

pos_count = max(int(np.sum(y_train == 1)), 1)
neg_count = max(int(np.sum(y_train == 0)), 1)
pos_weight = len(y_train) / (2.0 * pos_count)
neg_weight = len(y_train) / (2.0 * neg_count)

best_state = copy.deepcopy(model.state_dict())
best_val_loss = float("inf")
no_improve = 0
best_epoch = 0

for epoch in range(MAX_EPOCHS):
    model.train()
    train_probs = model(X_train_t)
    train_losses = bce(train_probs, y_train_t)
    train_weights = torch.where(
        y_train_t > 0.5,
        torch.full_like(y_train_t, pos_weight),
        torch.full_like(y_train_t, neg_weight),
    )
    train_loss = (train_losses * train_weights).mean()

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_probs_t = model(X_val_t)
        val_losses = bce(val_probs_t, y_val_t)
        val_weights = torch.where(
            y_val_t > 0.5,
            torch.full_like(y_val_t, pos_weight),
            torch.full_like(y_val_t, neg_weight),
        )
        val_loss = (val_losses * val_weights).mean().item()

    if val_loss < best_val_loss - EARLY_STOP_MIN_DELTA:
        best_val_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())
        no_improve = 0
        best_epoch = epoch
    else:
        no_improve += 1

    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Train Loss: {train_loss.item():.4f} | Val Loss: {val_loss:.4f}")

    if no_improve >= EARLY_STOP_PATIENCE:
        print(f"Early stopping at epoch {epoch} (best epoch {best_epoch})")
        break

model.load_state_dict(best_state)

val_probs_raw = infer_probs(model, X_val_t)
test_probs_raw = infer_probs(model, X_test_t)

calibrator = None
val_probs_cal = val_probs_raw.copy()
test_probs_cal = test_probs_raw.copy()

if len(np.unique(y_val)) > 1:
    calibrator = LogisticRegression(random_state=SEED, solver="lbfgs")
    calibrator.fit(val_probs_raw.reshape(-1, 1), y_val)
    val_probs_cal = calibrator.predict_proba(val_probs_raw.reshape(-1, 1))[:, 1]
    test_probs_cal = calibrator.predict_proba(test_probs_raw.reshape(-1, 1))[:, 1]

best_val_threshold, val_sweep = choose_threshold(y_val, val_probs_cal)
selected_threshold = best_val_threshold["threshold"]

test_metrics_selected = threshold_metrics(y_test, test_probs_cal, selected_threshold)
test_metrics_default = threshold_metrics(y_test, test_probs_cal, 0.5)
test_auc = safe_auc(y_test, test_probs_cal)
brier = float(np.mean((test_probs_cal - y_test) ** 2))

print("\n===== THRESHOLD SWEEP (Validation, calibrated) =====")
for t in [0.40, 0.45, 0.50, 0.55, 0.60]:
    row = min(val_sweep, key=lambda m: abs(m["threshold"] - t))
    print_row("VAL ", row)
print_row("VAL*", best_val_threshold)
print(
    f"Selected threshold: {selected_threshold:.2f} "
    f"(min_recall={MIN_HUMAN_RECALL:.2f}, metric=max_f1)"
)

print("\n===== MODEL EVALUATION (Test, calibrated) =====")
print_row("T@0.50", test_metrics_default)
print_row("T@best", test_metrics_selected)
print(f"AUC: {test_auc:.3f}")
print(f"Brier score: {brier:.4f}")

torch.save(model.state_dict(), MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

if calibrator is not None:
    joblib.dump(calibrator, CALIBRATOR_PATH)
elif os.path.exists(CALIBRATOR_PATH):
    os.remove(CALIBRATOR_PATH)

with open(THRESHOLD_PATH, "w", encoding="utf-8") as f:
    json.dump(
        {
            "threshold": float(selected_threshold),
            "min_human_recall": float(MIN_HUMAN_RECALL),
            "validation_best": best_val_threshold,
            "test_metrics_selected": test_metrics_selected,
            "test_metrics_default": test_metrics_default,
            "test_auc": float(test_auc),
            "test_brier": float(brier),
            "calibrated": bool(calibrator is not None),
        },
        f,
        indent=2,
    )

print("\nTraining complete.")
print("Model saved as luma_model.pth")
print("Scaler saved as scaler.pkl")
print("Calibrator saved as calibrator.pkl" if calibrator is not None else "Calibrator not saved")
print("Threshold saved as threshold.json")
