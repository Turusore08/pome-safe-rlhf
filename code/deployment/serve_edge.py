"""
Local Edge Inference Service with FastAPI and ONNXRuntime.
Deploys the safe RLHF controller directly at the photobioreactor facility.
Implements def serve() and def predict() for deterministic edge actuation.
(penjelasan.md §6).
"""

import os
import sys
import numpy as np
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import onnxruntime as ort

app = FastAPI(title="POME RLHF Edge Controller", version="1.0.0")

# Global session holder
SESSION: Optional[ort.InferenceSession] = None
MODEL_PATH = "deployment/models/policy_edge.onnx"


class SensorPayload(BaseModel):
    state_vector: List[float]


class ActionResponse(BaseModel):
    action: List[float]
    aeration_hours: float
    sodium_bicarbonate_grams: float
    microalgae_inoculum_ml: float
    molasses_ml: float
    light_intensity_lux: float


def init_session(model_path: str = MODEL_PATH) -> ort.InferenceSession:
    """Initialize ONNXRuntime InferenceSession with CPU optimizations."""
    global SESSION
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"[Strict Error] Model ONNX tidak ditemukan di '{model_path}'. "
            f"Penggunaan model dummy/fallback dilarang sesuai protokol riset. "
            f"Latih kebijakan pada data riil dan jalankan: python deployment/export_onnx.py terlebih dahulu."
        )

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    SESSION = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
    return SESSION


@app.on_event("startup")
def startup_event():
    init_session()


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "pome-edge-inference"}


@app.post("/predict", response_model=ActionResponse)
def predict(payload: SensorPayload):
    """
    Edge inference endpoint:
    Consumes fused 224-dim observation vector and predicts 5 continuous dosing levers.
    """
    global SESSION
    if SESSION is None:
        init_session()

    input_data = np.array(payload.state_vector, dtype=np.float32).reshape(1, -1)
    if input_data.shape[1] != 224:
        raise HTTPException(
            status_code=400,
            detail=f"Expected state_vector of dimension 224, got {input_data.shape[1]}",
        )

    input_name = SESSION.get_inputs()[0].name
    output_name = SESSION.get_outputs()[0].name

    raw_action = SESSION.run([output_name], {input_name: input_data})[0][0]
    acts = [float(x) for x in raw_action]

    return ActionResponse(
        action=acts,
        aeration_hours=acts[0],
        sodium_bicarbonate_grams=acts[1],
        microalgae_inoculum_ml=acts[2],
        molasses_ml=acts[3],
        light_intensity_lux=acts[4],
    )


def serve(host: str = "0.0.0.0", port: int = 8000):
    """Launch local edge inference server via uvicorn."""
    import uvicorn
    print(f"Starting POME edge controller service on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    init_session()
    print("Edge inference session initialized successfully.")
