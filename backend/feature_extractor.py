"""
Feature Extractor — ResNet50 backbone for image feature extraction.
Extracted from page.py training pipeline for inference use.
"""

import torch
import torch.nn as nn
from torchvision import models


class FeatureExtractor(nn.Module):
    """
    Extracts features from images using a pre-trained ResNet50.
    Output: 2048-dimensional feature vector per image.
    """

    def __init__(self, model_name: str = "resnet50", weights_path: str | None = None):
        super().__init__()

        if model_name == "resnet50":
            # Build architecture without pretrained weights first
            base_model = models.resnet50(weights=None)
            self.features = nn.Sequential(*list(base_model.children())[:-1])
            self.output_dim = 2048

        elif model_name == "vgg16":
            base_model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            self.features = base_model.features
            self.avgpool = base_model.avgpool
            self.output_dim = 512 * 7 * 7

        elif model_name == "densenet121":
            base_model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
            self.features = base_model.features
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.output_dim = 1024

        elif model_name == "mobilenet_v2":
            base_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
            self.features = base_model.features
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.output_dim = 1280
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Load saved weights (from training environment) if provided
        if weights_path is not None:
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            self.load_state_dict(state_dict)

        # Freeze all weights — inference only
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        if hasattr(self, "avgpool"):
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # Flatten to (batch, features)
        return x
