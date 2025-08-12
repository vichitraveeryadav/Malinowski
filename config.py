import os
from pathlib import Path

# Where files are stored
BASE_DIR = Path.cwd()
UPLOAD_DIR = BASE_DIR / "uploads"
PROCESSED_DIR = BASE_DIR / "processed"
DATABASE_URL = "sqlite:///immigration_docs.db"

# Types of documents we can recognize
DOCUMENT_TYPES = {
    "passport": ["passport", "travel document"],
    "visa": ["visa", "entry permit"], 
    "permit": ["work permit", "study permit"],
    "certificate": ["birth certificate", "marriage certificate"],
    "identification": ["driver license", "id card"]
}

# Languages for OCR (text reading)
OCR_LANGUAGES_TESSERACT = ["eng", "hin"]  # English and Hindi for Tesseract
OCR_LANGUAGES_EASYOCR = ["en", "hi"]      # English and Hindi for EasyOCR
