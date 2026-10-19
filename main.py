"""
AnemoScan Inference Service
----------------------------
A small, standalone prediction API for binary anemia detection.

The service takes a conjunctiva image, runs it through the binary ONNX
classifier, and returns the probability of anemia plus a binary decision.

Endpoints:
  GET  /health   -> quick check that the service and model are up
  POST /predict  -> multipart file upload, returns
                    {anemiaDetected, anemiaProbability, status}
"""

import io
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

MODEL_PATH = Path(__file__).resolve().parent / "model" / "model.onnx"
IMAGE_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

app = FastAPI(title="AnemoScan Inference Service")

# Allow requests from the main AnemoScan app. This is intentionally open for
# the student project; it can be restricted to the exact deployed frontend
# URL later if needed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_session: ort.InferenceSession | None = None


def get_session() -> ort.InferenceSession:
    global _session
    if _session is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=f"model.onnx not found at {MODEL_PATH}. Deploy it alongside this service.",
            )
        _session = ort.InferenceSession(
            str(MODEL_PATH),
            providers=["CPUExecutionProvider"],
        )
    return _session


def sigmoid(value: float) -> float:
    # Numerically stable enough for a single classifier logit.
    if value >= 0:
        return 1.0 / (1.0 + np.exp(-value))
    exp_value = np.exp(value)
    return exp_value / (1.0 + exp_value)


def preprocess_image(raw_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)

    array = np.asarray(image, dtype=np.float32) / 255.0  # HWC, [0,1]
    array = (array - IMAGENET_MEAN) / IMAGENET_STD  # ImageNet normalization
    array = array.transpose(2, 0, 1)  # HWC -> CHW
    array = np.expand_dims(array, axis=0)  # (1, 3, 224, 224)
    return array.astype(np.float32)


@app.get("/health")
def health():
    model_present = MODEL_PATH.exists()
    return {"status": "ok", "model_present": model_present}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    session = get_session()

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Empty file upload.")

    try:
        input_array = preprocess_image(raw_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read this as an image.")

    try:
        results = session.run(None, {"input": input_array})
        anemia_logit = float(results[0][0])
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Model inference failed: {exc}",
        )

    anemia_probability = float(sigmoid(anemia_logit))
    anemia_detected = bool(anemia_probability >= 0.5)

    return {
        "anemiaDetected": anemia_detected,
        "anemiaProbability": round(anemia_probability, 4),
        "status": "Anemia Detected" if anemia_detected else "No Anemia Detected",
    }
