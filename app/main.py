from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from enum import Enum
import shutil
import os
from app.ocr import parse_layout_and_ocr

app = FastAPI(title="Docling + Tesseract OCR Parser")

class OutputFormat(str, Enum):
    json = "json"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/upload/")
async def upload_invoice(
    file: UploadFile = File(...),
    output_format: OutputFormat = Form(OutputFormat.json)
):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = parse_layout_and_ocr(file_path)
    return JSONResponse(content=result)
