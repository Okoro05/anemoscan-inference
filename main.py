"""
AnemoScan Inference Service
----------------------------
A small, standalone prediction API — deployed separately from the main
Netlify app, on Render (a real persistent server, not a size-capped
serverless function). Takes a conjunctiva image, runs it through the
trained ONNX model, and returns a predicted hemoglobin value + status.

Endpoints:
  GET  /health   -> quick check that the service and model are up
  POST /predict  -> multipart file upload, returns {predictedHb, status}
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

# Allow requests from your Netlify site (and anywhere, for simplicity as a
# student project). Tighten allow_origins to your exact site URL later if
# you want to lock this down.
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
        _session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    return _session


def classify_severity(hb: float) -> str:
    if hb >= 12:
        return "Normal"
    if hb >= 10:
        return "Mild Anemia"
    if hb >= 7:
        return "Moderate Anemia"
    return "Severe Anemia"


def preprocess_image(raw_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)

    array = np.asarray(image, dtype=np.float32) / 255.0  # HWC, [0,1]
    array = (array - IMAGENET_MEAN) / IMAGENET_STD  # normalize
    array = array.transpose(2, 0, 1)  # HWC -> CHW
    array = np.expand_dims(array, axis=0)  # add batch dim -> (1, 3, 224, 224)
    return array.astype(np.float32)


@app.get("/health")
def health():
    model_loaded = MODEL_PATH.exists()
    return {"status": "ok", "model_present": model_loaded}


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

    results = session.run(None, {"input": input_array})
    predicted_hb = float(results[0][0])

    return {
        "predictedHb": round(predicted_hb, 2),
        "status": classify_severity(predicted_hb),
    }
