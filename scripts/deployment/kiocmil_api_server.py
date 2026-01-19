"""
FastAPI Server for KIOCMIL-CADA Model

Provides endpoints for knee OA grading using KIOCMIL-CADA model.
Requires 3 models:
- Knee detection model
- Lesion detection model
- KIOCMIL-CADA classification model
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from pydantic import BaseModel
from typing import Optional, List, Dict
import uvicorn
import cv2
import numpy as np

from src.api.kiocmil_inference import KiocmilInference
from src.api.utils import read_image_file, encode_image_base64

app = FastAPI(
    title="KIOCMIL-CADA API",
    description="Knee Osteoarthritis Grading with Context-Aware Deformable Attention MIL",
)

kiocmil_pipeline = None


class KneePrediction(BaseModel):
    knee_bbox: List[int]
    predicted_class: str
    predicted_class_id: int
    confidence: float
    class_probabilities: Dict[str, float]
    num_js_lesions: int
    num_ost_lesions: int


class KiocmilResponse(BaseModel):
    filename: str
    predictions: List[KneePrediction]
    image_base64: Optional[str] = None


def load_pipeline(
    kiocmil_model: str,
    knee_model: str,
    lesion_model: str,
    num_classes: int = 10,
):
    """Initialize KIOCMIL inference pipeline."""
    global kiocmil_pipeline
    kiocmil_pipeline = KiocmilInference(
        kiocmil_model_path=kiocmil_model,
        knee_model_path=knee_model,
        lesion_model_path=lesion_model,
        num_classes=num_classes,
    )


@app.post("/predict", response_model=KiocmilResponse)
async def predict(
    file: UploadFile = File(...),
    knee_conf: float = Form(0.25),
    lesion_conf: float = Form(0.25),
    return_image: bool = Form(False),
):
    """
    Predict knee OA grade using KIOCMIL-CADA model.

    Args:
        file: Input X-ray image
        knee_conf: Confidence threshold for knee detection
        lesion_conf: Confidence threshold for lesion detection
        return_image: Whether to return annotated image

    Returns:
        Predictions for each detected knee
    """
    global kiocmil_pipeline
    if kiocmil_pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")

    # Read image
    contents = await file.read()
    image_rgb = read_image_file(contents, file.filename)

    # Run inference
    results = kiocmil_pipeline.predict(
        image_rgb,
        knee_conf=knee_conf,
        lesion_conf=lesion_conf,
    )

    # Format response
    predictions = [KneePrediction(**r) for r in results]

    # Optionally create annotated image
    encoded_image = None
    if return_image and predictions:
        vis_img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        for pred in predictions:
            x1, y1, x2, y2 = pred.knee_bbox

            # Draw knee bbox
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label
            label = f"{pred.predicted_class} ({pred.confidence:.2f})"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

            # Background for text
            cv2.rectangle(
                vis_img,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                (0, 255, 0),
                -1,
            )
            cv2.putText(
                vis_img,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )

            # Add lesion count
            info = f"JS:{pred.num_js_lesions} OST:{pred.num_ost_lesions}"
            cv2.putText(
                vis_img,
                info,
                (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        encoded_image = encode_image_base64(vis_img)

    return KiocmilResponse(
        filename=file.filename,
        predictions=predictions,
        image_base64=encoded_image,
    )


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "pipeline_loaded": kiocmil_pipeline is not None,
        "model_type": "KIOCMIL-CADA",
    }


@app.get("/model_info")
def model_info():
    """Get model information."""
    if kiocmil_pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")

    return {
        "model_type": "KIOCMIL-CADA",
        "num_classes": kiocmil_pipeline.num_classes,
        "ctx_size": kiocmil_pipeline.ctx_size,
        "patch_size": kiocmil_pipeline.patch_size,
        "device": str(kiocmil_pipeline.device),
    }


def start_kiocmil_server(
    host: str,
    port: int,
    kiocmil_model: str,
    knee_model: str,
    lesion_model: str,
    num_classes: int = 10,
):
    """
    Start KIOCMIL-CADA API server.

    Args:
        host: Host to bind
        port: Port to bind
        kiocmil_model: Path to KIOCMIL-CADA checkpoint
        knee_model: Path to knee detection model
        lesion_model: Path to lesion detection model
        num_classes: Number of output classes
    """
    load_pipeline(kiocmil_model, knee_model, lesion_model, num_classes)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="KIOCMIL-CADA API Server")
    parser.add_argument(
        "--kiocmil-model",
        type=str,
        default="runs/kiocmil_cada/cada_10class_unbalanced/best_acc_model.pt",
        help="Path to KIOCMIL-CADA model",
    )
    parser.add_argument(
        "--knee-model",
        type=str,
        default="runs/detect/my_knee_run_resplit/weights/best.pt",
        help="Path to knee detection model",
    )
    parser.add_argument(
        "--lesion-model",
        type=str,
        default="runs/detect/my_knee_run_resplit/weights/best.pt",
        help="Path to lesion detection model",
    )
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)

    args = parser.parse_args()

    start_kiocmil_server(
        args.host,
        args.port,
        args.kiocmil_model,
        args.knee_model,
        args.lesion_model,
        args.num_classes,
    )
