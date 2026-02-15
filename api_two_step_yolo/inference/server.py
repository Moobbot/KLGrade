from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, Response
import uvicorn
import cv2
import numpy as np
import io
import os
from typing import Optional
from pathlib import Path

# Import the inference class
from api_two_step_yolo.inference.two_step_yolo_api import TwoStepYOLOInference

app = FastAPI(
    title="KL Grade Prediction API",
    description="API for Two-Step Knee Detection & Lesion Analysis using YOLO11l",
    version="1.0.0",
)

# Initialize pipeline
PIPELINE = None
KNEE_MODEL_PATH = "runs/detect/knee_detector/weights/best.pt"
LESION_MODEL_PATH = "runs/detect/lesion_8class_balanced/weights/best.pt"


@app.on_event("startup")
async def load_model():
    global PIPELINE
    try:
        if not os.path.exists(KNEE_MODEL_PATH):
            raise FileNotFoundError(f"Knee model not found at {KNEE_MODEL_PATH}")
        if not os.path.exists(LESION_MODEL_PATH):
            raise FileNotFoundError(f"Lesion model not found at {LESION_MODEL_PATH}")

        print(f"Loading models...")
        PIPELINE = TwoStepYOLOInference(
            knee_model_path=KNEE_MODEL_PATH,
            lesion_model_path=LESION_MODEL_PATH,
            device="cuda:0",
        )
        print("Models loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {e}")
        pass


@app.post("/predict/")
async def predict(file: UploadFile = File(...), visualize: bool = False):
    """
    Predict KL Grade from an uploaded X-ray image.
    """
    if PIPELINE is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        if visualize:
            # We can't use visualize() directly because it expects path to load for cv2.imread
            # We need to adapt visualizer or just implement simple viz here or save temp
            # For simplicity, saving temp is fine for visualization
            temp_path = f"temp_{file.filename}"
            cv2.imwrite(temp_path, image)

            viz_path = f"viz_{file.filename}"
            PIPELINE.visualize(temp_path, viz_path)

            with open(viz_path, "rb") as f:
                img_bytes = f.read()

            os.remove(temp_path)
            os.remove(viz_path)
            return Response(content=img_bytes, media_type="image/jpeg")

        else:
            # Pass numpy array directly (Thanks to refactor)
            result = PIPELINE.predict(image)

            # Convert numpy types to native python types
            json_result = {
                "filename": file.filename,
                "kl_grade": (
                    int(result["kl_grade"]) if result["kl_grade"] is not None else None
                ),
                "knees_count": len(result["knees"]),
                "lesions_count": len(result["lesions"]),
                "knees": [
                    {
                        "bbox": [int(x) for x in k["bbox"]],
                        "confidence": float(k["confidence"]),
                        "knee_id": int(k["knee_id"]),
                    }
                    for k in result["knees"]
                ],
                "lesions": [
                    {
                        "bbox": [int(x) for x in l["bbox"]],
                        "class_name": l["class_name"],
                        "confidence": float(l["confidence"]),
                        "knee_id": int(l["knee_id"]),
                    }
                    for l in result["lesions"]
                ],
            }
            return JSONResponse(content=json_result)

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9090)
