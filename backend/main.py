"""
Backend API — FastAPI inference server for casting defect classification.
Pipeline: Image → ResNet50 feature extraction (GPU) → StandardScaler → SVM prediction
"""

import io
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from joblib import load
from PIL import Image
from scipy.spatial.distance import cdist
from torchvision import transforms

from feature_extractor import FeatureExtractor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_DIR = Path("/models")
SVM_MODEL_PATH = MODEL_DIR / "svm_model.joblib"
SCALER_PATH = MODEL_DIR / "scaler.joblib"
RESNET_WEIGHTS_PATH = MODEL_DIR / "resnet50_extractor.pth"
FEATURES_NPZ_PATH = MODEL_DIR / "features_dataset.npz"
SIMILARITY_CONFIG_PATH = MODEL_DIR / "similarity_config.json"
CASTING_DATA_DIR = Path("/casting_data")
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Classes — binary classification matching training order
# Training: ok_front → label 0, def_front → label 1
CLASSES = ["ok", "def"]  # index 0 = ok, index 1 = defective

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backend")

# ---------------------------------------------------------------------------
# Global model references (loaded at startup)
# ---------------------------------------------------------------------------
extractor: FeatureExtractor | None = None
svm_model = None
scaler = None

# Similarity search data (loaded from ia_training output)
dataset_features: np.ndarray | None = None
dataset_paths: np.ndarray | None = None
dataset_labels: np.ndarray | None = None
similarity_config: dict | None = None

# Image transform — must match training (test_transform from page.py)
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ---------------------------------------------------------------------------
# Startup / Shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models into memory at startup."""
    global extractor, svm_model, scaler, dataset_features, dataset_paths, dataset_labels, similarity_config

    logger.info("=" * 60)
    logger.info("LOADING MODELS")
    logger.info("=" * 60)

    # 1. Feature extractor (ResNet50) — load exact weights from training
    t0 = time.time()
    if not RESNET_WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"ResNet50 weights not found: {RESNET_WEIGHTS_PATH}. "
            "Please save them from the training notebook."
        )
    extractor = FeatureExtractor(
        model_name="resnet50",
        weights_path=str(RESNET_WEIGHTS_PATH),
    )
    extractor = extractor.to(DEVICE)
    extractor.eval()
    logger.info(f"  ResNet50 loaded from {RESNET_WEIGHTS_PATH} on {DEVICE} in {time.time() - t0:.2f}s")

    # 2. SVM model
    if not SVM_MODEL_PATH.exists():
        raise FileNotFoundError(f"SVM model not found: {SVM_MODEL_PATH}")
    svm_model = load(SVM_MODEL_PATH)
    logger.info(f"  SVM model loaded from {SVM_MODEL_PATH}")

    # 3. Scaler (StandardScaler)
    if SCALER_PATH.exists():
        scaler = load(SCALER_PATH)
        logger.info(f"  Scaler loaded from {SCALER_PATH}")
    else:
        scaler = None
        logger.warning(
            "  ⚠️  Scaler not found! Predictions may be inaccurate. "
            "Please save scaler.joblib from training."
        )

    # 4. Similarity dataset (features_dataset.npz from ia_training)
    if FEATURES_NPZ_PATH.exists():
        data = np.load(FEATURES_NPZ_PATH, allow_pickle=True)
        dataset_features = data["features"]
        dataset_paths = data["paths"]
        dataset_labels = data["labels"]
        logger.info(
            f"  Similarity dataset loaded: {dataset_features.shape[0]} images, "
            f"{dataset_features.shape[1]}-dim features"
        )
    else:
        logger.warning(
            f"  ⚠️  Similarity dataset not found: {FEATURES_NPZ_PATH}. "
            "Run ia_training.ipynb first. /api/similar will be unavailable."
        )

    # 5. Similarity config (metric choice from ia_training)
    if SIMILARITY_CONFIG_PATH.exists():
        with open(SIMILARITY_CONFIG_PATH) as f:
            similarity_config = json.load(f)
        logger.info(f"  Similarity config: {similarity_config}")
    else:
        similarity_config = {"type": "single", "metric": "cosine"}
        logger.warning(
            "  ⚠️  Similarity config not found, defaulting to cosine distance"
        )

    logger.info("=" * 60)
    logger.info("MODELS READY — accepting requests")
    logger.info("=" * 60)

    yield  # App is running

    # Cleanup
    logger.info("Shutting down, releasing models...")


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Casting Defect Classifier API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "device": str(DEVICE),
        "cuda_available": torch.cuda.is_available(),
        "svm_loaded": svm_model is not None,
        "scaler_loaded": scaler is not None,
    }


@app.post("/api/classify")
async def classify(file: UploadFile = File(...)):
    """
    Classify a casting image as OK or Defective.

    Accepts: image file (JPEG/PNG)
    Returns: { label, label_fr, confidence, inference_time_ms }
    """
    if extractor is None or svm_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    # Validate file type
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        t_start = time.time()

        # 1. Read image
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 2. Transform
        img_tensor = test_transform(img).unsqueeze(0).to(DEVICE)

        # 3. Extract features with ResNet50
        with torch.no_grad():
            features = extractor(img_tensor).cpu().numpy()

        # 4. Scale features (if scaler available)
        if scaler is not None:
            features_scaled = scaler.transform(features)
        else:
            features_scaled = features

        # 5. SVM prediction
        prediction = svm_model.predict(features_scaled)[0]
        label = CLASSES[int(prediction)]

        # Try to get decision function for confidence
        try:
            decision = svm_model.decision_function(features_scaled)[0]
            confidence = float(1 / (1 + np.exp(-abs(decision))))  # Sigmoid approx
        except Exception:
            confidence = 1.0

        inference_time = (time.time() - t_start) * 1000  # ms

        # Human-readable labels
        label_map = {
            "ok": {"label": "ok", "label_fr": "Pièce Conforme ✅", "color": "#22c55e"},
            "def": {"label": "def", "label_fr": "Pièce Défectueuse ❌", "color": "#ef4444"},
        }
        result = label_map.get(label, label_map["def"])

        logger.info(
            f"Classified '{file.filename}' → {label} "
            f"(conf: {confidence:.2f}, {inference_time:.0f}ms)"
        )

        return JSONResponse({
            **result,
            "confidence": round(confidence, 3),
            "inference_time_ms": round(inference_time, 1),
            "filename": file.filename,
        })

    except Exception as e:
        logger.error(f"Classification error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------
def _normalize_distances(distances: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]."""
    dmin, dmax = distances.min(), distances.max()
    if dmax - dmin < 1e-10:
        return np.zeros_like(distances)
    return (distances - dmin) / (dmax - dmin)


def compute_distances(query_vec: np.ndarray, db: np.ndarray, config: dict) -> np.ndarray:
    """Compute distances according to similarity_config."""
    q = query_vec.reshape(1, -1)

    if config.get("type") == "combined":
        weights = config["weights"]
        combined = np.zeros(db.shape[0])
        for metric_name, weight in weights.items():
            raw = cdist(q, db, metric=metric_name).flatten()
            combined += weight * _normalize_distances(raw)
        return combined
    else:
        metric = config.get("metric", "cosine")
        return cdist(q, db, metric=metric).flatten()


# ---------------------------------------------------------------------------
# Similarity endpoint
# ---------------------------------------------------------------------------
@app.post("/api/similar")
async def find_similar(file: UploadFile = File(...)):
    """
    Find the 10 most similar images in the dataset.
    Also returns classification result.
    """
    if extractor is None or svm_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    if dataset_features is None:
        raise HTTPException(
            status_code=503,
            detail="Similarity dataset not loaded. Run ia_training.ipynb first.",
        )

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        t_start = time.time()

        # 1. Read & transform image
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_tensor = test_transform(img).unsqueeze(0).to(DEVICE)

        # 2. Extract features
        with torch.no_grad():
            features = extractor(img_tensor).cpu().numpy()

        # 3. Scale
        if scaler is not None:
            features_scaled = scaler.transform(features)
        else:
            features_scaled = features

        # 4. Classification (reuse same pipeline)
        prediction = svm_model.predict(features_scaled)[0]
        label = CLASSES[int(prediction)]
        try:
            decision = svm_model.decision_function(features_scaled)[0]
            confidence = float(1 / (1 + np.exp(-abs(decision))))
        except Exception:
            confidence = 1.0

        label_map = {
            "ok": {"label": "ok", "label_fr": "Pièce Conforme ✅", "color": "#22c55e"},
            "def": {"label": "def", "label_fr": "Pièce Défectueuse ❌", "color": "#ef4444"},
        }
        result_info = label_map.get(label, label_map["def"])

        # 5. Find top-10 similar images
        query_vec = features_scaled.flatten()
        distances = compute_distances(query_vec, dataset_features, similarity_config or {})
        top_indices = np.argsort(distances)[:10]

        similar = []
        for rank, idx in enumerate(top_indices):
            similar.append({
                "rank": rank + 1,
                "path": str(dataset_paths[idx]),
                "label": str(dataset_labels[idx]),
                "distance": round(float(distances[idx]), 6),
                "image_url": f"/api/images/{dataset_paths[idx]}",
            })

        inference_time = (time.time() - t_start) * 1000

        logger.info(
            f"Similar search '{file.filename}' → {label} "
            f"(conf: {confidence:.2f}, {inference_time:.0f}ms)"
        )

        return JSONResponse({
            **result_info,
            "confidence": round(confidence, 3),
            "inference_time_ms": round(inference_time, 1),
            "filename": file.filename,
            "metric": (similarity_config or {}).get("name", "cosine"),
            "similar": similar,
        })

    except Exception as e:
        logger.error(f"Similarity search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Similarity search failed: {str(e)}")


# ---------------------------------------------------------------------------
# Serve dataset images
# ---------------------------------------------------------------------------
@app.get("/api/images/{path:path}")
async def serve_image(path: str):
    """Serve an image from the casting_data directory."""
    # Security: prevent path traversal
    clean_path = Path(path)
    if ".." in clean_path.parts:
        raise HTTPException(status_code=400, detail="Invalid path")

    full_path = CASTING_DATA_DIR / clean_path
    if not full_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    # Ensure it's actually inside casting_data
    try:
        full_path.resolve().relative_to(CASTING_DATA_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid path")

    return FileResponse(full_path, media_type="image/jpeg")
