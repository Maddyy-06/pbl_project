import os
from pathlib import Path

class Config:
    IMG_SIZE = 128
    # Clinical HU Windowing (The "Kaggle Standard")
    HU_MIN = -150
    HU_MAX = 250
    CROP_SIZE = (128, 128)
    
    # Path Logic
    BASE_DIR = Path(__file__).resolve().parent
    YOLO_MODEL = str(BASE_DIR / "models" / "yolo_final.pt")
    UNET_MODEL = str(BASE_DIR / "models" / "unet_tumor_final.pth")
    SAMPLE_CASES = BASE_DIR / "datasets" / "sample_cases"