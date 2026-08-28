import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent


class Config:
    FLASK_DEBUG = os.getenv("FLASK_DEBUG", "1") == "1"
    BACKEND_HOST = os.getenv("BACKEND_HOST", "127.0.0.1")
    BACKEND_PORT = int(os.getenv("BACKEND_PORT", "5000"))
    MODEL_PATH = Path(os.getenv("MODEL_PATH", PROJECT_ROOT / "models" / "visionarc-efficientnet.pt"))
    UPLOAD_FOLDER = Path(os.getenv("UPLOAD_FOLDER", PROJECT_ROOT / "data" / "raw"))
    CORS_ORIGINS = [
        origin.strip()
        for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")
        if origin.strip()
    ]
