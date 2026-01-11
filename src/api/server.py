from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from pydantic import BaseModel
from typing import Optional
from src.api.inference import YOLOModel
from src.api.utils import read_image_file, encode_image_base64
import uvicorn

app = FastAPI(title="YOLO Inference API", description="API for Knee Osteoarthritis Detection")
model_instance = None

class Prediction(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: dict

class InferenceResponse(BaseModel):
    filename: str
    predictions: list[Prediction]
    image_base64: Optional[str] = None

def load_model(model_path: str):
    global model_instance
    model_instance = YOLOModel(model_path)

@app.post("/predict", response_model=InferenceResponse)
async def predict(
    file: UploadFile = File(...),
    conf: float = Form(0.25),
    return_image: bool = Form(False)
):
    global model_instance
    if model_instance is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    contents = await file.read()
    image_rgb = read_image_file(contents, file.filename)
    
    # Run Inference
    results = model_instance.predict(image_rgb, conf=conf)
    
    if not results:
        return InferenceResponse(filename=file.filename, predictions=[])
        
    result = results[0]
    predictions = []
    
    # Extract boxes
    boxes = result.boxes
    for i in range(len(boxes)):
        box = boxes[i]
        xyxy = box.xyxy[0].cpu().numpy().tolist()
        cls_id = int(box.cls[0].item())
        conf_score = float(box.conf[0].item())
        
        # Use config mapping if available, fallback to model names
        # YOLOModel handled mapping resolution, but result logic uses internal IDs.
        # So we map cls_id -> mapped name
        class_name = model_instance.class_mapping.get(cls_id, str(cls_id))
        
        predictions.append(Prediction(
            class_id=cls_id,
            class_name=class_name,
            confidence=conf_score,
            bbox={
                "x1": xyxy[0],
                "y1": xyxy[1],
                "x2": xyxy[2],
                "y2": xyxy[3]
            }
        ))
        
    encoded_image = None
    if return_image:
        annotated_bgr = result.plot()
        encoded_image = encode_image_base64(annotated_bgr)
        
    return InferenceResponse(
        filename=file.filename,
        predictions=predictions,
        image_base64=encoded_image
    )

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model_instance is not None}

def start_api_server(host: str, port: int, model_path: str):
    load_model(model_path)
    uvicorn.run(app, host=host, port=port)
