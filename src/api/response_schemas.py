"""
Response Schemas for KLGrade API

Standardized Pydantic models for all API responses with comprehensive documentation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from enum import Enum


class StatusEnum(str, Enum):
    """API response status."""

    SUCCESS = "success"
    ERROR = "error"


class KLGradeClass(str, Enum):
    """KL Grade classification classes (from config.py CLASSES_10_CLASS)."""

    KL0_A = "KL0-a"  # Osteophyte (gai xương)
    KL0_B = "KL0-b"  # Joint space (khe khớp)
    KL1_A = "KL1-a"
    KL1_B = "KL1-b"
    KL2_A = "KL2-a"
    KL2_B = "KL2-b"
    KL3_A = "KL3-a"
    KL3_B = "KL3-b"
    KL4_A = "KL4-a"
    KL4_B = "KL4-b"


class BoundingBox(BaseModel):
    """Bounding box coordinates."""

    x1: int = Field(..., description="Top-left x coordinate")
    y1: int = Field(..., description="Top-left y coordinate")
    x2: int = Field(..., description="Bottom-right x coordinate")
    y2: int = Field(..., description="Bottom-right y coordinate")

    class Config:
        schema_extra = {"example": {"x1": 100, "y1": 150, "x2": 300, "y2": 450}}


class LesionDetail(BaseModel):
    """
    Detailed information for a single detected lesion.

    Each lesion represents an individual pathological finding (osteophyte or joint space narrowing)
    detected by the lesion detection model.
    """

    lesion_type: str = Field(
        ...,
        description=(
            "Type of lesion detected: "
            "'js' = Joint space narrowing (khe khớp hẹp), "
            "'ost' = Osteophyte (gai xương)"
        ),
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Detection confidence score for this specific lesion (0.0-1.0)",
    )
    bbox: BoundingBox = Field(
        ..., description="Bounding box coordinates of the lesion in the original image"
    )


class LesionsSummary(BaseModel):
    """
    Summary statistics of all detected lesions in a knee.

    Provides quick overview of lesion counts by type.
    """

    js: int = Field(
        ..., ge=0, description="Total number of joint space (JS) lesions detected"
    )
    ost: int = Field(
        ..., ge=0, description="Total number of osteophyte (OST) lesions detected"
    )
    total: int = Field(..., ge=0, description="Total number of all lesions (js + ost)")


class KneePrediction(BaseModel):
    """
    Prediction for a single knee region.

    This contains the overall KL grade classification for the knee,
    along with detailed information about individual lesions detected.

    Fields explanation:
    - knee_bbox: Location of the knee in the image
    - knee_confidence: How confident the detector is that this is a knee
    - predicted_class: Final KL grade (e.g., KL2-a means Grade 2 with Osteophyte dominance)
    - confidence: How confident the model is in this classification
    - class_probabilities: Probability distribution showing confidence for each possible grade
    - lesions_summary: Quick count of detected lesions
    - lesions_by_class: Detailed list of each lesion, grouped by their detected class
    """

    knee_bbox: BoundingBox = Field(
        ..., description="Bounding box of the detected knee region in the image"
    )
    knee_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score for knee detection (0.0-1.0)"
    )
    predicted_class: str = Field(
        ...,
        description=(
            "Overall predicted KL grade class for this knee. "
            "This is the final classification based on all detected lesions. "
            "Format: KLX-Y where X=grade (0-4), Y=type (a=Osteophyte, b=Joint space)"
        ),
    )
    predicted_class_id: int = Field(
        ...,
        description="Numeric ID of the predicted class (0-9 mapping to KL0-a through KL4-b)",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Confidence score for the overall knee classification (0.0-1.0). "
            "This represents how confident the model is in the predicted_class."
        ),
    )
    class_probabilities: Dict[str, float] = Field(
        ...,
        description=(
            "Probability distribution over all possible KL grade classes. "
            "Shows the model's confidence for each class (KL0-a through KL4-b). "
            "All probabilities sum to 1.0."
        ),
    )
    lesions_summary: LesionsSummary = Field(
        ...,
        description=(
            "Summary statistics of detected lesions. "
            "Provides quick counts of joint space (js) and osteophyte (ost) lesions."
        ),
    )
    lesions_by_class: Dict[str, List[LesionDetail]] = Field(
        ...,
        description=(
            "Detailed list of all detected lesions, grouped by their detected class. "
            "Each lesion includes its type (js/ost), detection confidence score, and bounding box. "
            "Multiple classes may be present if lesions of different grades are detected. "
            "Key = class name (e.g., 'KL2-a'), Value = list of lesions detected for that class."
        ),
    )

    class Config:
        schema_extra = {
            "example": {
                "knee_bbox": {"x1": 100, "y1": 150, "x2": 300, "y2": 450},
                "knee_confidence": 0.92,
                "predicted_class": "KL2-a",
                "predicted_class_id": 4,
                "confidence": 0.87,
                "class_probabilities": {
                    "KL0-a": 0.01,
                    "KL0-b": 0.02,
                    "KL1-a": 0.05,
                    "KL1-b": 0.03,
                    "KL2-a": 0.87,
                    "KL2-b": 0.01,
                    "KL3-a": 0.005,
                    "KL3-b": 0.003,
                    "KL4-a": 0.001,
                    "KL4-b": 0.001,
                },
                "lesions_summary": {"js": 4, "ost": 3, "total": 7},
                "lesions_by_class": {
                    "KL2-a": [
                        {
                            "lesion_type": "ost",
                            "score": 0.91,
                            "bbox": {"x1": 110, "y1": 180, "x2": 140, "y2": 210},
                        },
                        {
                            "lesion_type": "ost",
                            "score": 0.84,
                            "bbox": {"x1": 240, "y1": 350, "x2": 270, "y2": 380},
                        },
                    ],
                    "KL2-b": [
                        {
                            "lesion_type": "js",
                            "score": 0.88,
                            "bbox": {"x1": 120, "y1": 200, "x2": 150, "y2": 230},
                        },
                        {
                            "lesion_type": "js",
                            "score": 0.85,
                            "bbox": {"x1": 160, "y1": 250, "x2": 190, "y2": 280},
                        },
                        {
                            "lesion_type": "js",
                            "score": 0.77,
                            "bbox": {"x1": 200, "y1": 300, "x2": 230, "y2": 330},
                        },
                    ],
                    "KL1-a": [
                        {
                            "lesion_type": "ost",
                            "score": 0.72,
                            "bbox": {"x1": 130, "y1": 220, "x2": 155, "y2": 245},
                        }
                    ],
                    "KL1-b": [
                        {
                            "lesion_type": "js",
                            "score": 0.68,
                            "bbox": {"x1": 180, "y1": 280, "x2": 210, "y2": 310},
                        }
                    ],
                },
            }
        }


class PredictionResponse(BaseModel):
    """Standard prediction response."""

    status: StatusEnum = Field(
        default=StatusEnum.SUCCESS, description="Response status"
    )
    filename: str = Field(..., description="Name of the uploaded file")
    num_knees_detected: int = Field(
        ..., ge=0, description="Number of knees detected in image"
    )
    predictions: List[KneePrediction] = Field(
        ..., description="List of predictions for each detected knee"
    )
    processing_time_ms: float = Field(
        ..., description="Processing time in milliseconds"
    )

    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "filename": "knee_xray.jpg",
                "num_knees_detected": 2,
                "predictions": [
                    {
                        "knee_bbox": {"x1": 100, "y1": 150, "x2": 300, "y2": 450},
                        "knee_confidence": 0.92,
                        "predicted_class": "KL2-a",
                        "predicted_class_id": 4,
                        "confidence": 0.87,
                        "class_probabilities": {},
                        "lesions_summary": {"js": 4, "ost": 3, "total": 7},
                        "lesions_by_class": {},
                    }
                ],
                "processing_time_ms": 234.5,
            }
        }


class VisualResponse(BaseModel):
    """Response with visual annotations."""

    status: StatusEnum = Field(
        default=StatusEnum.SUCCESS, description="Response status"
    )
    filename: str = Field(..., description="Name of the uploaded file")
    num_knees_detected: int = Field(
        ..., ge=0, description="Number of knees detected in image"
    )
    predictions: List[KneePrediction] = Field(
        ..., description="List of predictions for each detected knee"
    )
    annotated_image_base64: Optional[str] = Field(
        None, description="Base64 encoded annotated image with bounding boxes"
    )
    gradcam_image_base64: Optional[str] = Field(
        None, description="Base64 encoded GradCAM heatmap overlay"
    )
    processing_time_ms: float = Field(
        ..., description="Processing time in milliseconds"
    )

    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "filename": "knee_xray.jpg",
                "num_knees_detected": 2,
                "predictions": [],
                "annotated_image_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
                "gradcam_image_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
                "processing_time_ms": 456.7,
            }
        }


class ErrorDetail(BaseModel):
    """Error detail information."""

    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[str] = Field(None, description="Additional error details")


class ErrorResponse(BaseModel):
    """Standard error response."""

    status: StatusEnum = Field(default=StatusEnum.ERROR, description="Response status")
    error: ErrorDetail = Field(..., description="Error details")

    class Config:
        schema_extra = {
            "example": {
                "status": "error",
                "error": {
                    "code": "INVALID_IMAGE",
                    "message": "Failed to decode image file",
                    "details": "Unsupported image format",
                },
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    pipeline_loaded: bool = Field(
        ..., description="Whether the model pipeline is loaded"
    )
    model_type: str = Field(..., description="Type of model loaded")


class ModelInfoResponse(BaseModel):
    """Model information response."""

    model_type: str = Field(..., description="Model architecture type")
    num_classes: int = Field(..., description="Number of output classes")
    ctx_size: List[int] = Field(..., description="Context patch size [height, width]")
    patch_size: List[int] = Field(..., description="Lesion patch size [height, width]")
    device: str = Field(..., description="Device used for inference (cpu/cuda)")
    class_names: List[str] = Field(..., description="List of class names")

    class Config:
        schema_extra = {
            "example": {
                "model_type": "KIOCMIL-CADA",
                "num_classes": 10,
                "ctx_size": [384, 384],
                "patch_size": [224, 224],
                "device": "cuda",
                "class_names": [
                    "KL0-a",
                    "KL0-b",
                    "KL1-a",
                    "KL1-b",
                    "KL2-a",
                    "KL2-b",
                    "KL3-a",
                    "KL3-b",
                    "KL4-a",
                    "KL4-b",
                ],
            }
        }
