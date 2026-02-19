from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, Response
import uvicorn
import cv2
import base64
import numpy as np
import io
import os
from typing import Optional
from pathlib import Path

# Import the inference class
from api_two_step_yolo.inference.two_step_yolo_api import TwoStepYOLOInference
from src.api.utils import read_image_file

app = FastAPI(
    title="KL Grade Prediction API",
    description="""
    Two-Step Deep Learning Pipeline for Knee Osteoarthritis Grading.
    
    ## Features
    - **Step 1: Knee Detection** (YOLO11n) - Detects knee joints in X-ray images.
    - **Step 2: Lesion Analysis** (YOLO11l) - Identifies osteophytes and JSN to determine KL Grade.
    
    ## Usage
    - **Standard X-ray**: Upload an image to `/predict/` to get predictions.
    - **DICOM**: Upload a `.dcm` file to `/predict/dicom/` to get predictions and optional visualization.
    """,
    version="2.0.0",
    terms_of_service="http://example.com/terms/",
    contact={
        "name": "KLGrade Team",
        "url": "http://github.com/ngoductam/KLGrade",
        "email": "tam@example.com",
    },
    license_info={
        "name": "MIT",
    },
)

# Initialize pipeline
PIPELINE = None
# Best YOLO11N Knee Detector (99.5% mAP)
KNEE_MODEL_PATH = "runs/detect/knee_yolo11n_20260217_134003/weights/best.pt"
# Best Lesion Detector (8-Class Balanced)
LESION_MODEL_PATH = "runs/detect/lesion_8class_balanced/weights/best.pt"


@app.on_event("startup")
async def load_model():
    global PIPELINE
    try:
        if not os.path.exists(KNEE_MODEL_PATH):
            print(
                f"Warning: Knee model not found at {KNEE_MODEL_PATH}. Checking env vars..."
            )
        if not os.path.exists(LESION_MODEL_PATH):
            print(
                f"Warning: Lesion model not found at {LESION_MODEL_PATH}. Checking env vars..."
            )

        print(f"Loading models...")
        PIPELINE = TwoStepYOLOInference(
            knee_model_path=os.getenv("KNEE_MODEL", KNEE_MODEL_PATH),
            lesion_model_path=os.getenv("LESION_MODEL", LESION_MODEL_PATH),
            device="cuda:0",
        )
        print("Models loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {e}")
        pass


@app.post(
    "/predict/",
    tags=["Inference"],
    summary="Predict KL Grade",
    response_description="JSON response containing predicted KL grade, knee bounding boxes, lesion details, and optional base64 visualization.",
)
async def predict(file: UploadFile = File(...), visualize: bool = False):
    """
    **Upload an X-ray image** to detect knees and classify Osteoarthritis severity (KL Grade).

    - **file**: Input X-ray image (JPEG/PNG)
    - **visualize**: If true, returns JSON results including a base64 encoded annotated image in the `visualization` field.
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

        # Predict
        result = PIPELINE.predict(image)

        # Format initial response
        json_result = {
            "filename": file.filename,
            "kl_grade": (
                int(result["kl_grade"]) if result["kl_grade"] is not None else None
            ),
            "image_size": result["image_size"],
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
                    "bbox": [int(x) for x in l.get("bbox_global", l["bbox"])],
                    "class_name": l["class_name"],
                    "confidence": float(l["confidence"]),
                    "knee_id": int(l["knee_id"]),
                }
                for l in result["lesions"]
            ],
        }

        if visualize:
            # Draw on a copy of the image (use BGR for cv2 drawing)
            viz_image = image.copy()

            # Draw knee boxes (Green)
            for knee in result["knees"]:
                x1, y1, x2, y2 = [int(v) for v in knee["bbox"]]
                cv2.rectangle(viz_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(
                    viz_image,
                    f"Knee {knee['knee_id']}: {knee['confidence']:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            # Draw lesion boxes (Red)
            for lesion in result["lesions"]:
                # Use bbox_global if available, else bbox
                bbox = lesion.get("bbox_global", lesion["bbox"])
                x1, y1, x2, y2 = [int(v) for v in bbox]
                cv2.rectangle(viz_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    viz_image,
                    f"{lesion['class_name']}: {lesion['confidence']:.2f}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )

            # Draw KL grade (Blue)
            kl_grade = result["kl_grade"]
            if kl_grade is not None:
                cv2.putText(
                    viz_image,
                    f"KL Grade: {kl_grade}",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (255, 0, 0),
                    3,
                )

            # Encode to base64
            _, buffer = cv2.imencode(".png", viz_image)
            viz_b64 = base64.b64encode(buffer).decode("utf-8")

            json_result["visualization"] = viz_b64

        return JSONResponse(content=json_result)

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/predict/dicom/",
    tags=["Inference"],
    summary="Predict KL Grade from DICOM",
    response_description="JSON response containing predicted KL grade, knee bounding boxes, lesion details, and optional base64 visualization.",
)
async def predict_dicom(
    file: UploadFile = File(..., description="DICOM file to be processed"),
    visualize: bool = False,
):
    """
    **Upload a DICOM file** to detect knees and classify Osteoarthritis severity (KL Grade).

    - **file**: Input DICOM file (.dcm)
    - **visualize**: If true, returns JSON results including a base64 encoded annotated image in the `visualization` field.
    """
    if PIPELINE is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        contents = await file.read()
        # read_image_file returns RGB
        image_rgb = read_image_file(contents, file.filename)

        # Convert to BGR as PIPELINE seems to expect BGR (based on server.py usage of cv2.imdecode)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Predict
        result = PIPELINE.predict(image_bgr)

        # Format initial response
        json_result = {
            "filename": file.filename,
            "kl_grade": (
                int(result["kl_grade"]) if result["kl_grade"] is not None else None
            ),
            "image_size": result["image_size"],
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
                    "bbox": [int(x) for x in l.get("bbox_global", l["bbox"])],
                    "class_name": l["class_name"],
                    "confidence": float(l["confidence"]),
                    "knee_id": int(l["knee_id"]),
                }
                for l in result["lesions"]
            ],
        }

        if visualize:
            # Draw on a copy of the image (use BGR for cv2 drawing)
            viz_image = image_bgr.copy()

            # Draw knee boxes (Green)
            for knee in result["knees"]:
                x1, y1, x2, y2 = [int(v) for v in knee["bbox"]]
                cv2.rectangle(viz_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(
                    viz_image,
                    f"Knee {knee['knee_id']}: {knee['confidence']:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            # Draw lesion boxes (Red)
            for lesion in result["lesions"]:
                # Use bbox_global if available, else bbox
                bbox = lesion.get("bbox_global", lesion["bbox"])
                x1, y1, x2, y2 = [int(v) for v in bbox]
                cv2.rectangle(viz_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    viz_image,
                    f"{lesion['class_name']}: {lesion['confidence']:.2f}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )

            # Draw KL grade (Blue)
            kl_grade = result["kl_grade"]
            if kl_grade is not None:
                cv2.putText(
                    viz_image,
                    f"KL Grade: {kl_grade}",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (255, 0, 0),
                    3,
                )

            # Encode to base64
            _, buffer = cv2.imencode(".png", viz_image)
            viz_b64 = base64.b64encode(buffer).decode("utf-8")

            json_result["visualization"] = viz_b64

        return JSONResponse(content=json_result)

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9090)
