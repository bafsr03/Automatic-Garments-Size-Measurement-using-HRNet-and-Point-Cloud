import os
import io
import shutil
import uuid
from typing import Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from .pipeline import process_image_request


app = FastAPI(title="Garment Measurement API")


@app.post("/process")
async def process(image: UploadFile = File(...)) -> Dict[str, Any]:
    if image.content_type is None or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file")

    request_id = str(uuid.uuid4())
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    work_dir = os.path.join(base_dir, "api_runs", request_id)
    os.makedirs(work_dir, exist_ok=True)

    input_image_path = os.path.join(work_dir, "image.jpg")
    with open(input_image_path, "wb") as f:
        content = await image.read()
        f.write(content)

    try:
        result = process_image_request(input_image_path, work_dir)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse(content=result)


@app.get("/files")
def get_file(path: str):
    abs_path = os.path.abspath(path)
    # allow only within repo for safety
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if not abs_path.startswith(base_dir):
        raise HTTPException(status_code=403, detail="Access denied")
    if not os.path.exists(abs_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(abs_path)


