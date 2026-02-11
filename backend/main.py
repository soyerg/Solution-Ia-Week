"""
Backend API — FastAPI inference server for casting defect classification.
Pipeline: Image → ResNet50 feature extraction (GPU) → StandardScaler → SVM prediction
"""

import io
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from joblib import load
from PIL import Image
from torchvision import transforms

from feature_extractor import FeatureExtractor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_DIR = Path("/models")
SVM_MODEL_PATH = MODEL_DIR / "svm_model.joblib"
SCALER_PATH = MODEL_DIR / "scaler.joblib"
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Classes — binary classification matching training order
CLASSES = ["def", "ok"]  # index 0 = defective, index 1 = ok

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backend")

# ---------------------------------------------------------------------------
# Global model references (loaded at startup)
# ---------------------------------------------------------------------------
extractor: FeatureExtractor | None = None
svm_model = None
scaler = None

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
    global extractor, svm_model, scaler

    logger.info("=" * 60)
    logger.info("LOADING MODELS")
    logger.info("=" * 60)

    # 1. Feature extractor (ResNet50)
    t0 = time.time()
    extractor = FeatureExtractor(model_name="resnet50")
    extractor = extractor.to(DEVICE)
    extractor.eval()
    logger.info(f"  ResNet50 loaded on {DEVICE} in {time.time() - t0:.2f}s")

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
