"""
FastAPI Server for KIOCMIL-CADA Model

Enhanced API with:
- Prediction endpoint (bounding boxes + classifications)
- Visual endpoint (annotated images + GradCAM)
- Comprehensive logging
- API documentation (Swagger UI)
- Standardized response formats
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import uvicorn
import cv2
import numpy as np
import time
from typing import Optional

from src.api.kiocmil_inference import KiocmilInference
from src.api.utils import read_image_file, encode_image_base64
from src.api.response_schemas import (
    PredictionResponse,
    VisualResponse,
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    KneePrediction,
    BoundingBox,
    StatusEnum,
    ErrorDetail,
)
from src.api.logger_config import (
    setup_logger,
    log_request,
    log_response,
    log_error,
    log_model_load,
    log_inference,
)
from src.api.gradcam import (
    generate_multi_scale_attention,
    create_heatmap_overlay,
    draw_annotations,
)

# Initialize logger
logger = setup_logger("klgrade_api", log_dir="logs")

# Initialize FastAPI app with metadata
app = FastAPI(
    title="KLGrade KIOCMIL-CADA API",
    description="""
    ## Knee Osteoarthritis Grading API
    
    Advanced API for automated KL grading using KIOCMIL-CADA (Context-Aware Deformable Attention) architecture.
    
    ### Features:
    - **Multi-knee detection**: Automatically detects and analyzes multiple knees in X-ray images
    - **10-class classification**: Fine-grained classification (Healthy, Doubtful/Minimal/Moderate/Severe for JS and OST)
    - **Lesion detection**: Identifies joint space narrowing and osteophyte lesions
    - **Visual explanations**: GradCAM attention maps showing model focus areas
    
    ### Endpoints:
    - `POST /predict`: Get predictions with bounding boxes
    - `POST /predict_visual`: Get predictions with annotated images and GradCAM
    - `GET /health`: Health check
    - `GET /model_info`: Model information
    """,
    version="2.0.0",
    contact={
        "name": "KLGrade Team",
        "email": "https://www.linkedin.com/in/duc-tam-ngo-29306a215",
    },
    license_info={"name": "MIT License"},
)

kiocmil_pipeline = None

# Class names mapping (from config.py CLASSES_10_CLASS)
CLASS_NAMES_10 = [
    "KL0-a",  # Osteophyte (gai xương)
    "KL0-b",  # Joint space (khe khớp)
    "KL1-a",
    "KL1-b",
    "KL2-a",
    "KL2-b",
    "KL3-a",
    "KL3-b",
    "KL4-a",
    "KL4-b",
]


def load_pipeline(
    kiocmil_model: str,
    knee_model: str,
    lesion_model: str,
    num_classes: int = 10,
):
    """Initialize KIOCMIL inference pipeline."""
    global kiocmil_pipeline

    try:
        log_model_load(logger, "KIOCMIL-CADA", kiocmil_model)
        log_model_load(logger, "Knee Detection", knee_model)
        log_model_load(logger, "Lesion Detection", lesion_model)

        kiocmil_pipeline = KiocmilInference(
            kiocmil_model_path=kiocmil_model,
            knee_model_path=knee_model,
            lesion_model_path=lesion_model,
            num_classes=num_classes,
        )

        logger.info("✅ Pipeline loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}", exc_info=True)
        raise


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        200: {"description": "Successful prediction"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Server error"},
    },
    summary="Predict KL Grade",
    description="Analyze knee X-ray image and return predictions with bounding boxes",
)
async def predict(
    file: UploadFile = File(..., description="X-ray image file (JPEG, PNG, or DICOM)"),
    knee_conf: float = Form(
        0.5, ge=0.0, le=1.0, description="Confidence threshold for knee detection"
    ),
    lesion_conf: float = Form(
        0.5, ge=0.0, le=1.0, description="Confidence threshold for lesion detection"
    ),
):
    """
    Predict KL grade from knee X-ray image.

    Returns predictions with bounding boxes for each detected knee.
    """
    start_time = time.time()

    try:
        # Validate pipeline
        if kiocmil_pipeline is None:
            raise HTTPException(status_code=500, detail="Pipeline not initialized")

        # Log request
        log_request(
            logger,
            "/predict",
            file.filename,
            {"knee_conf": knee_conf, "lesion_conf": lesion_conf},
        )

        # Read image
        contents = await file.read()
        image_rgb = read_image_file(contents, file.filename)

        # Run inference
        results = kiocmil_pipeline.predict(
            image_rgb,
            knee_conf=knee_conf,
            lesion_conf=lesion_conf,
        )

        # Format predictions
        predictions = []
        total_js = 0
        total_ost = 0

        for result in results:
            # Convert bbox tuple to BoundingBox model
            bbox_tuple = result["knee_bbox"]
            bbox = BoundingBox(
                x1=int(bbox_tuple[0]),
                y1=int(bbox_tuple[1]),
                x2=int(bbox_tuple[2]),
                y2=int(bbox_tuple[3]),
            )

            # Convert lesion bboxes
            js_bboxes = [
                BoundingBox(x1=int(b[0]), y1=int(b[1]), x2=int(b[2]), y2=int(b[3]))
                for b in result.get("js_lesion_bboxes", [])
            ]
            ost_bboxes = [
                BoundingBox(x1=int(b[0]), y1=int(b[1]), x2=int(b[2]), y2=int(b[3]))
                for b in result.get("ost_lesion_bboxes", [])
            ]

            prediction = KneePrediction(
                knee_bbox=bbox,
                predicted_class=result["predicted_class"],
                predicted_class_id=result["predicted_class_id"],
                confidence=result["confidence"],
                class_probabilities=result["class_probabilities"],
                num_js_lesions=result["num_js_lesions"],
                num_ost_lesions=result["num_ost_lesions"],
                js_lesion_bboxes=js_bboxes,
                ost_lesion_bboxes=ost_bboxes,
            )
            predictions.append(prediction)

            total_js += result["num_js_lesions"]
            total_ost += result["num_ost_lesions"]

        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000

        # Log inference metrics
        log_inference(logger, len(predictions), total_js, total_ost, processing_time)

        # Create response
        response = PredictionResponse(
            status=StatusEnum.SUCCESS,
            filename=file.filename,
            num_knees_detected=len(predictions),
            predictions=predictions,
            processing_time_ms=processing_time,
        )

        # Log response
        log_response(logger, "/predict", "success", processing_time)

        return response

    except HTTPException:
        raise
    except Exception as e:
        log_error(logger, "/predict", e)
        error_response = ErrorResponse(
            status=StatusEnum.ERROR,
            error=ErrorDetail(
                code="PREDICTION_ERROR",
                message="Failed to process image",
                details=str(e),
            ),
        )
        return JSONResponse(status_code=500, content=error_response.dict())


@app.post(
    "/predict_visual",
    response_model=VisualResponse,
    responses={
        200: {"description": "Successful prediction with visualizations"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Server error"},
    },
    summary="Predict with Visualizations",
    description="Analyze knee X-ray and return predictions with annotated images and GradCAM heatmaps",
)
async def predict_visual(
    file: UploadFile = File(..., description="X-ray image file (JPEG, PNG, or DICOM)"),
    knee_conf: float = Form(
        0.5, ge=0.0, le=1.0, description="Confidence threshold for knee detection"
    ),
    lesion_conf: float = Form(
        0.5, ge=0.0, le=1.0, description="Confidence threshold for lesion detection"
    ),
    include_gradcam: bool = Form(
        True, description="Whether to include GradCAM heatmap"
    ),
):
    """
    Predict KL grade with visual explanations.

    Returns:
    - Predictions with bounding boxes
    - Annotated image with bounding boxes and labels
    - GradCAM attention heatmap (optional)
    """
    start_time = time.time()

    try:
        # Validate pipeline
        if kiocmil_pipeline is None:
            raise HTTPException(status_code=500, detail="Pipeline not initialized")

        # Log request
        log_request(
            logger,
            "/predict_visual",
            file.filename,
            {
                "knee_conf": knee_conf,
                "lesion_conf": lesion_conf,
                "include_gradcam": include_gradcam,
            },
        )

        # Read image
        contents = await file.read()
        image_rgb = read_image_file(contents, file.filename)

        # Run inference
        results = kiocmil_pipeline.predict(
            image_rgb,
            knee_conf=knee_conf,
            lesion_conf=lesion_conf,
        )

        # Format predictions
        predictions = []
        knee_bboxes = []
        total_js = 0
        total_ost = 0

        for result in results:
            bbox_tuple = result["knee_bbox"]
            bbox = BoundingBox(
                x1=int(bbox_tuple[0]),
                y1=int(bbox_tuple[1]),
                x2=int(bbox_tuple[2]),
                y2=int(bbox_tuple[3]),
            )

            # Convert lesion bboxes
            js_bboxes = [
                BoundingBox(x1=int(b[0]), y1=int(b[1]), x2=int(b[2]), y2=int(b[3]))
                for b in result.get("js_lesion_bboxes", [])
            ]
            ost_bboxes = [
                BoundingBox(x1=int(b[0]), y1=int(b[1]), x2=int(b[2]), y2=int(b[3]))
                for b in result.get("ost_lesion_bboxes", [])
            ]

            prediction = KneePrediction(
                knee_bbox=bbox,
                predicted_class=result["predicted_class"],
                predicted_class_id=result["predicted_class_id"],
                confidence=result["confidence"],
                class_probabilities=result["class_probabilities"],
                num_js_lesions=result["num_js_lesions"],
                num_ost_lesions=result["num_ost_lesions"],
                js_lesion_bboxes=js_bboxes,
                ost_lesion_bboxes=ost_bboxes,
            )
            predictions.append(prediction)
            knee_bboxes.append(bbox_tuple)

            total_js += result["num_js_lesions"]
            total_ost += result["num_ost_lesions"]

        # Generate annotated image
        annotated_image = draw_annotations(
            image_rgb, results, show_confidence=True, show_lesion_count=True
        )

        # Convert to BGR for encoding
        annotated_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        annotated_base64 = encode_image_base64(annotated_bgr)

        # Generate GradCAM if requested
        gradcam_base64 = None
        if include_gradcam and len(knee_bboxes) > 0:
            try:
                # Generate attention maps
                attention_maps = generate_multi_scale_attention(
                    kiocmil_pipeline.model,
                    image_rgb,
                    None,  # batch_data not needed for simple attention
                    knee_bboxes,
                )

                # Combine attention maps
                combined_attention = (
                    np.maximum.reduce(attention_maps)
                    if attention_maps
                    else np.zeros(image_rgb.shape[:2])
                )

                # Create heatmap overlay
                gradcam_overlay = create_heatmap_overlay(
                    image_rgb, combined_attention, alpha=0.5
                )

                # Convert to BGR and encode
                gradcam_bgr = cv2.cvtColor(gradcam_overlay, cv2.COLOR_RGB2BGR)
                gradcam_base64 = encode_image_base64(gradcam_bgr)

                logger.info("GradCAM heatmap generated successfully")
            except Exception as e:
                logger.warning(f"Failed to generate GradCAM: {e}")

        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000

        # Log inference metrics
        log_inference(logger, len(predictions), total_js, total_ost, processing_time)

        # Create response
        response = VisualResponse(
            status=StatusEnum.SUCCESS,
            filename=file.filename,
            num_knees_detected=len(predictions),
            predictions=predictions,
            annotated_image_base64=annotated_base64,
            gradcam_image_base64=gradcam_base64,
            processing_time_ms=processing_time,
        )

        # Log response
        log_response(logger, "/predict_visual", "success", processing_time)

        return response

    except HTTPException:
        raise
    except Exception as e:
        log_error(logger, "/predict_visual", e)
        error_response = ErrorResponse(
            status=StatusEnum.ERROR,
            error=ErrorDetail(
                code="VISUALIZATION_ERROR",
                message="Failed to generate visualizations",
                details=str(e),
            ),
        )
        return JSONResponse(status_code=500, content=error_response.dict())


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check if the API service and model pipeline are running",
)
def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        pipeline_loaded=kiocmil_pipeline is not None,
        model_type="KIOCMIL-CADA",
    )


@app.get(
    "/model_info",
    response_model=ModelInfoResponse,
    summary="Model Information",
    description="Get detailed information about the loaded model",
)
def model_info():
    """Get model information."""
    if kiocmil_pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")

    return ModelInfoResponse(
        model_type="KIOCMIL-CADA",
        num_classes=kiocmil_pipeline.num_classes,
        ctx_size=list(kiocmil_pipeline.ctx_size),
        patch_size=list(kiocmil_pipeline.patch_size),
        device=str(kiocmil_pipeline.device),
        class_names=CLASS_NAMES_10[: kiocmil_pipeline.num_classes],
    )


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
    logger.info(f"Starting KIOCMIL-CADA API Server on {host}:{port}")
    load_pipeline(kiocmil_model, knee_model, lesion_model, num_classes)
    logger.info(f"API Documentation available at http://{host}:{port}/docs")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="KIOCMIL-CADA API Server")
    parser.add_argument(
        "--kiocmil-model",
        type=str,
        default="runs/kiocmil_cada/cada_10class_balanced/best_acc_model.pt",
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
