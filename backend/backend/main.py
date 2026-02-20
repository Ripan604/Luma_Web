from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
import hashlib
import joblib
from fastapi.middleware.cors import CORSMiddleware
import os
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "luma_model.pth")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
DATASET_PATH = os.path.join(BASE_DIR, "dataset.json")

# ===============================
# CLASSIFIER MODEL
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

# ===============================
# LOAD MODEL (SAFE LOADING)
# ===============================

scaler = None
model = None
input_size = 103  # default

if os.path.exists(SCALER_PATH) and os.path.exists(MODEL_PATH):
    scaler = joblib.load(SCALER_PATH)
    input_size = scaler.mean_.shape[0]

    model = LumaFusionModel(input_size).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

# ===============================
# REQUEST MODEL
# ===============================

class SessionData(BaseModel):
    session_vector: list
    content: str
    label: int | None = None  # Only used for collect

# ===============================
# VERIFY ENDPOINT
# ===============================

@app.post("/verify")
def verify(data: SessionData):

    if model is None or scaler is None:
        raise HTTPException(status_code=400, detail="Model not trained yet.")

    if len(data.session_vector) != input_size:
        raise HTTPException(
            status_code=400,
            detail=f"Expected vector length {input_size}"
        )

    session_vector = np.array(data.session_vector, dtype=np.float32)

    # Clip typing count
    session_vector[-1] = np.clip(session_vector[-1], 0, 300)

    session_vector = session_vector.reshape(1, -1)

    # Apply scaling
    session_vector_scaled = scaler.transform(session_vector)

    tensor = torch.tensor(session_vector_scaled, dtype=torch.float32).to(device)

    with torch.no_grad():
        prob = model(tensor).item()

    prediction = 1 if prob >= 0.5 else 0

    combined_hash = hashlib.sha256(
        data.content.encode() + session_vector.tobytes()
    ).hexdigest()

    return {
        "human_probability": float(prob),
        "prediction": prediction,
        "hash": combined_hash
    }

# ===============================
# COLLECT DATA
# ===============================

@app.post("/collect")
def collect(data: SessionData):

    if data.label is None:
        raise HTTPException(status_code=400, detail="Label required (1=human, 0=bot)")

    entry = {
        "session_vector": data.session_vector,
        "label": data.label
    }

    # Create dataset file if missing
    if not os.path.exists(DATASET_PATH):
        with open(DATASET_PATH, "w", encoding="utf-8") as f:
            json.dump([], f)

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    dataset.append(entry)

    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f)

    return {"message": "Session stored successfully"}

# ===============================
# GENERATE BOT DATA
# ===============================

@app.post("/generate_bot_data")
def generate_bot_data(count: int = 30):

    if not os.path.exists(DATASET_PATH):
        raise HTTPException(status_code=400, detail="No dataset found")

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    human_vectors = [entry["session_vector"] for entry in dataset if entry["label"] == 1]

    if len(human_vectors) == 0:
        raise HTTPException(status_code=400, detail="No human samples available")

    for _ in range(count):
        base = np.array(human_vectors[np.random.randint(len(human_vectors))]).copy()

        # Softer structured noise
        base += np.random.normal(0, 0.015, len(base))

        # Softer typing perturbation
        base[-3] += np.random.normal(0, 0.03)
        base[-2] += np.random.normal(0, 0.02)
        base[-1] += np.random.randint(-6, 6)

        base[-1] = np.clip(base[-1], 0, 300)

        dataset.append({
            "session_vector": base.tolist(),
            "label": 0
        })

    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f)

    return {"message": f"{count} bot samples added"}


# ===============================
# DEMO: GET BOT VECTOR (for presentation)
# ===============================

def _predict_probability(vec_list: list) -> float | None:
    """Run vector through loaded model; return human probability or None if no model."""
    if model is None or scaler is None or len(vec_list) != input_size:
        return None
    arr = np.array(vec_list, dtype=np.float32).reshape(1, -1)
    arr[:, -1] = np.clip(arr[:, -1], 0, 300)
    arr = scaler.transform(arr)
    t = torch.tensor(arr, dtype=torch.float32).to(device)
    with torch.no_grad():
        return model(t).item()


def _make_demo_bot_vector(human_vectors: list, strong: bool = False) -> list:
    """Generate a single synthetic bot vector with noise. strong=True for more bot-like."""
    base = np.array(human_vectors[np.random.randint(len(human_vectors))]).copy()
    if strong:
        # Stronger noise so model reliably classifies as bot
        base += np.random.normal(0, 0.08, len(base))
        base[-3] += np.random.normal(0, 0.15)
        base[-2] += np.random.normal(0, 0.12)
        base[-1] += np.random.randint(-25, 25)
    else:
        base += np.random.normal(0, 0.015, len(base))
        base[-3] += np.random.normal(0, 0.03)
        base[-2] += np.random.normal(0, 0.02)
        base[-1] += np.random.randint(-6, 6)
    base[-1] = np.clip(base[-1], 0, 300)
    return base.tolist()


@app.get("/demo_bot_vector")
def get_demo_bot_vector():
    """Return a session vector that the model classifies as Bot (for demos)."""
    if not os.path.exists(DATASET_PATH):
        raise HTTPException(status_code=400, detail="No dataset found. Collect human samples first.")

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    human_vectors = [e["session_vector"] for e in dataset if e["label"] == 1]
    bot_vectors = [e["session_vector"] for e in dataset if e["label"] == 0]

    if len(human_vectors) == 0:
        raise HTTPException(status_code=400, detail="No human samples. Use 'Collect Human Training Sample' first.")

    # If model is loaded, return a vector that scores as bot (prob < 0.5)
    if model is not None and scaler is not None:
        # Try existing bot vectors first (they were used in training)
        for _ in range(min(20, max(len(bot_vectors), 1))):
            if bot_vectors:
                vec = bot_vectors[np.random.randint(len(bot_vectors))]
            else:
                vec = _make_demo_bot_vector(human_vectors, strong=False)
            prob = _predict_probability(vec)
            if prob is not None and prob < 0.5:
                return {"session_vector": vec}

        # Try stronger synthetic bot vectors
        for _ in range(30):
            vec = _make_demo_bot_vector(human_vectors, strong=True)
            prob = _predict_probability(vec)
            if prob is not None and prob < 0.5:
                return {"session_vector": vec}

        # Fallback: return vector with lowest prob among a few attempts
        best_vec = _make_demo_bot_vector(human_vectors, strong=True)
        best_prob = _predict_probability(best_vec) or 1.0
        for _ in range(15):
            cand = _make_demo_bot_vector(human_vectors, strong=True)
            p = _predict_probability(cand)
            if p is not None and p < best_prob:
                best_prob = p
                best_vec = cand
        return {"session_vector": best_vec}

    # No model: return any bot-like vector for display
    if bot_vectors:
        vec = bot_vectors[np.random.randint(len(bot_vectors))]
    else:
        vec = _make_demo_bot_vector(human_vectors, strong=True)
    return {"session_vector": vec}
