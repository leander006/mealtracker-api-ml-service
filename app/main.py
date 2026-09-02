"""
Serves the trained classifier as a REST API. This is what Spring Boot's
inference-gateway (inference package) will call internally.

Run:
    uvicorn app.main:app --reload --port 8000

Endpoint:
    POST /predict  (multipart/form-data, field "image")
    ->  {"foodLabel": "grilled_salmon", "confidence": 0.91}
"""

import json
import io
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0

MODEL_PATH = Path("./models/food_classifier.pt")
CLASSES_PATH = Path("./models/class_names.json")

app = FastAPI(title="meal-tracker ml-service")

# Loaded ONCE at startup, reused for every request. Loading the model
# fresh per-request would be extremely slow and wasteful.
_model = None
_class_names = None
_device = None

_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@app.on_event("startup")
def load_model():
    global _model, _class_names, _device

    if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
        raise RuntimeError(
            f"Model files not found at {MODEL_PATH} / {CLASSES_PATH}. "
            "Run training/prepare_data.py then training/train.py first."
        )

    _class_names = json.loads(CLASSES_PATH.read_text())
    _device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, len(_class_names))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=_device))
    model.to(_device)
    model.eval()

    _model = model
    print(f"Model loaded on {_device} with {len(_class_names)} classes")


@app.get("/health")
def health():
    return {"status": "ok", "classes_loaded": len(_class_names) if _class_names else 0}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file")

    tensor = _transform(img).unsqueeze(0).to(_device)

    with torch.no_grad():
        outputs = _model(tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_idx = probabilities.max(1)

    return {
        "foodLabel": _class_names[predicted_idx.item()],
        "confidence": round(confidence.item(), 4),
    }
