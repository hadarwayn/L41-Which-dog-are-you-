"""Inference module — predict dog breed for any image.

Usage:
    from src.inference.predict import predict_dog_breed, load_trained_model

    model = load_trained_model("resnet50")
    results = predict_dog_breed("photo.jpg", model)
    # [{"breed": "golden_retriever", "confidence": 0.73}, ...]
"""

from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image

from src.config import DEVICE, MODELS_DIR, TOP_K, NUM_CLASSES
from src.data.augmentation import get_inference_transforms
from src.data.dataset import get_breed_names
from src.models import get_model


def load_trained_model(
    model_name: str,
    device: torch.device = DEVICE,
) -> nn.Module:
    """Load a trained model from checkpoint.

    Args:
        model_name: Model name (e.g., 'resnet50').
        device: Device to load model on.

    Returns:
        Model in eval mode with loaded weights.
    """
    model = get_model(model_name, num_classes=NUM_CLASSES, pretrained=False)

    weight_path = MODELS_DIR / f"best_{model_name}.pth"
    if not weight_path.exists():
        raise FileNotFoundError(
            f"No trained weights found at {weight_path}. "
            "Train the model first."
        )

    model.load_state_dict(torch.load(weight_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_dog_breed(
    image_path: str,
    model: nn.Module,
    breed_names: list[str] = None,
    top_k: int = TOP_K,
    device: torch.device = DEVICE,
) -> list[dict]:
    """Predict dog breed for a single image.

    Step-by-step:
      1. Load image
      2. Resize to 224x224 (or 299x299)
      3. Normalize with ImageNet stats
      4. Feed through model
      5. Apply softmax → probability distribution
      6. Extract top-K breeds with confidence

    Args:
        image_path: Path to image file.
        model: Trained model in eval mode.
        breed_names: List of breed names (auto-loaded if None).
        top_k: Number of top predictions to return.
        device: Device for inference.

    Returns:
        List of {"breed": str, "confidence": float} sorted by confidence.
    """
    if breed_names is None:
        breed_names = get_breed_names()

    # 1. Load image
    img = Image.open(image_path).convert("RGB")

    # 2-3. Resize and normalize
    transform = get_inference_transforms()
    img_tensor = transform(img).unsqueeze(0).to(device)

    # 4. Forward pass
    outputs = model(img_tensor)
    if isinstance(outputs, tuple):
        outputs = outputs[0]

    # 5. Softmax → probabilities
    probs = torch.softmax(outputs, dim=1).squeeze()

    # 6. Top-K predictions
    top_probs, top_indices = probs.topk(top_k)

    results = []
    for i in range(top_k):
        idx = top_indices[i].item()
        breed = breed_names[idx] if idx < len(breed_names) else f"class_{idx}"
        results.append({
            "breed": breed,
            "confidence": round(top_probs[i].item(), 4),
        })

    return results


@torch.no_grad()
def predict_batch(
    image_paths: list[str],
    model: nn.Module,
    breed_names: list[str] = None,
    top_k: int = TOP_K,
    device: torch.device = DEVICE,
) -> list[dict]:
    """Predict dog breeds for multiple images.

    Args:
        image_paths: List of image file paths.
        model: Trained model in eval mode.
        breed_names: List of breed names.
        top_k: Top-K predictions per image.
        device: Device for inference.

    Returns:
        List of {"image": str, "predictions": [{"breed", "confidence"}]}.
    """
    results = []
    for path in image_paths:
        try:
            preds = predict_dog_breed(path, model, breed_names, top_k, device)
            results.append({
                "image": str(path),
                "predictions": preds,
            })
        except Exception as e:
            results.append({
                "image": str(path),
                "error": str(e),
                "predictions": [],
            })
    return results
