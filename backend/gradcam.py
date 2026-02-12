"""
Grad-CAM — Heatmap generation for ResNet50 classifier.
Ported from ia_classification.ipynb for server-side inference.

Generates an overlay image (50% original + 50% Jet heatmap) returned as base64 PNG.
"""

import base64
import io
import logging
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

logger = logging.getLogger("gradcam")

IMG_SIZE = 224


# ---------------------------------------------------------------------------
# ResNet50 Classifier (with Grad-CAM hooks) — matches training notebook
# ---------------------------------------------------------------------------
class ResNet50Classifier(nn.Module):
    """ResNet50 with frozen backbone and trainable classification head.
    Stores layer4 activations/gradients for Grad-CAM."""

    def __init__(self, num_classes: int = 2, weights_path: str | None = None):
        super().__init__()

        base_model = models.resnet50(weights=None)

        # Load saved weights — remap from FeatureExtractor format if needed
        if weights_path and os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)

            # Check if keys start with "features." (FeatureExtractor format)
            # or with layer names (ResNet50Classifier format)
            sample_key = next(iter(state_dict))
            if sample_key.startswith("features."):
                # Remap FeatureExtractor keys → standard ResNet50 keys
                model_dict = base_model.state_dict()
                child_names = [
                    "conv1", "bn1", "relu", "maxpool",
                    "layer1", "layer2", "layer3", "layer4", "avgpool",
                ]
                mapped = {}
                for key, value in state_dict.items():
                    if key.startswith("features."):
                        parts = key.split(".", 2)
                        idx = int(parts[1])
                        if idx < len(child_names):
                            new_key = (
                                child_names[idx] + "." + parts[2]
                                if len(parts) > 2
                                else child_names[idx]
                            )
                            if new_key in model_dict and model_dict[new_key].shape == value.shape:
                                mapped[new_key] = value
                base_model.load_state_dict(mapped, strict=False)
                logger.info(f"  Grad-CAM: remapped {len(mapped)} params from FeatureExtractor format")
            else:
                # Direct ResNet50Classifier state_dict — load into this module
                pass  # Handled after building the full model below

        # Backbone layers
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = base_model.avgpool

        # Freeze backbone
        for module in [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]:
            for p in module.parameters():
                p.requires_grad = False

        # Classification head (must match training architecture)
        self.classifier = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

        # Grad-CAM hooks
        self.gradients = None
        self.activations = None

        # If weights_path contains full classifier state_dict, load it now
        if weights_path and os.path.exists(weights_path):
            sample_key = next(iter(state_dict))
            if not sample_key.startswith("features."):
                self.load_state_dict(state_dict, strict=False)
                logger.info("  Grad-CAM: loaded full classifier state_dict")

    def save_gradient(self, grad):
        self.gradients = grad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Save activations for Grad-CAM
        self.activations = x
        if x.requires_grad:
            x.register_hook(self.save_gradient)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# Jet colormap (manual implementation — avoids matplotlib dependency)
# Produces the same result as matplotlib.cm.jet for values in [0, 1]
# ---------------------------------------------------------------------------
def _jet_colormap(value: np.ndarray) -> np.ndarray:
    """Apply Jet colormap to a [0,1] array. Returns (H, W, 3) float in [0,1]."""
    # Jet colormap: blue -> cyan -> green -> yellow -> red
    r = np.clip(1.5 - np.abs(4.0 * value - 3.0), 0, 1)
    g = np.clip(1.5 - np.abs(4.0 * value - 2.0), 0, 1)
    b = np.clip(1.5 - np.abs(4.0 * value - 1.0), 0, 1)
    return np.stack([r, g, b], axis=-1)


# ---------------------------------------------------------------------------
# Core Grad-CAM computation
# ---------------------------------------------------------------------------
def compute_gradcam(
    model: ResNet50Classifier,
    img_tensor: torch.Tensor,
    device: torch.device,
    target_class: int | None = None,
) -> tuple[np.ndarray, int, float]:
    """
    Compute Grad-CAM heatmap for a single image.

    Args:
        model: ResNet50Classifier with gradient hooks
        img_tensor: (3, 224, 224) normalized tensor
        device: torch device
        target_class: class to explain (None = predicted class)

    Returns:
        heatmap: (224, 224) array normalized [0, 1]
        pred_class: int
        confidence: float
    """
    model.eval()
    img = img_tensor.unsqueeze(0).to(device)
    img.requires_grad_(True)

    # Forward pass
    output = model(img)
    probs = torch.softmax(output, dim=1)
    pred_class = output.argmax(dim=1).item()
    confidence = probs[0, pred_class].item()

    if target_class is None:
        target_class = pred_class

    # Backward on target class
    model.zero_grad()
    output[0, target_class].backward()

    # Get gradients and activations from layer4
    gradients = model.gradients      # (1, 2048, 7, 7)
    activations = model.activations  # (1, 2048, 7, 7)

    # Channel weights = GAP of gradients
    weights = gradients.mean(dim=[2, 3], keepdim=True)  # (1, 2048, 1, 1)

    # Weighted combination + ReLU
    cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, 7, 7)
    cam = torch.relu(cam)

    # Upsample to 224×224
    cam = nn.functional.interpolate(
        cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False
    )
    cam = cam.squeeze().detach().cpu().numpy()

    # Normalize to [0, 1]
    if cam.max() > 0:
        cam = cam / cam.max()

    return cam, pred_class, confidence


# ---------------------------------------------------------------------------
# Generate overlay image as base64 PNG
# ---------------------------------------------------------------------------
def generate_gradcam_overlay(
    image_bytes: bytes,
    model: ResNet50Classifier,
    transform: transforms.Compose,
    device: torch.device,
) -> str | None:
    """
    Generate a Grad-CAM overlay image (50% original + 50% Jet heatmap).

    Args:
        image_bytes: raw image bytes (JPEG/PNG)
        model: loaded ResNet50Classifier
        transform: test_transform (resize + normalize)
        device: torch device

    Returns:
        base64-encoded PNG string, or None on failure
    """
    try:
        # Open and resize image
        img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_resized = img_pil.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img_resized).astype(np.float64) / 255.0  # (224, 224, 3)

        # Transform for model
        img_tensor = transform(img_pil)

        # Compute Grad-CAM heatmap
        heatmap, pred_class, confidence = compute_gradcam(model, img_tensor, device)

        # Apply Jet colormap to heatmap
        heatmap_colored = _jet_colormap(heatmap)  # (224, 224, 3) float [0,1]

        # Blend: 50% original + 50% heatmap (same as notebook)
        overlay = np.clip(0.5 * img_array + 0.5 * heatmap_colored, 0, 1)

        # Convert to PIL and encode as base64 PNG
        overlay_uint8 = (overlay * 255).astype(np.uint8)
        overlay_img = Image.fromarray(overlay_uint8)

        buffer = io.BytesIO()
        overlay_img.save(buffer, format="PNG", optimize=True)
        buffer.seek(0)

        return base64.b64encode(buffer.read()).decode("ascii")

    except Exception as e:
        logger.error(f"Grad-CAM generation failed: {e}", exc_info=True)
        return None
