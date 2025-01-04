from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Query
from typing import Annotated
from models.model_loader import load_model
from models.preprocessing import preprocessing
from services.predict_service import model_predict
import os
from io import BytesIO
from PIL import Image
import json
from fastapi.responses import StreamingResponse

router = APIRouter()

MODEL_FILE_NAME = "FSC147.pth"
MODEL_CHECKPOINT = os.path.join(os.path.dirname(__file__), "..", f'models/checkpoints/{MODEL_FILE_NAME}')

# Load model
model = load_model(model_path=MODEL_CHECKPOINT, base=MODEL_CHECKPOINT)
model.eval()

@router.post("/predict")
async def predict(
    image: Annotated[UploadFile, File(description="A Query Image")],
    boxes: Annotated[str | None, Form(description="Exampler: boxes [y1, x1, y2, x2]")] = None,
    ):

    # read Image -> to PIL image
    if image.size > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File size too large. Max size is 10 MB.")

    image_bytes = await image.read() 
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    # read bboxes
    if boxes:
        try:
            boxes = json.loads(boxes)
        except json.JSONDecodeError:
            return {"error": "Invalid bounding box format. Expected a JSON string"}
    else:
        boxes = []

    # Preprocessing
    sample, num_shot = preprocessing(image, boxes)

    y_density, count_predict = model_predict(model, sample, num_shot)

    # Convert to bytes
    img_density = BytesIO()
    y_density.save(img_density, format='PNG')
    img_density.seek(0)

    metadata = {
        "count_predict": count_predict
    } 

    return StreamingResponse(
        content=img_density,
        media_type="image/png",
        headers={"X-Metadata": json.dumps(metadata)},  # Send metadata in custom header
    )
           

if __name__ == "__main__":
    pass