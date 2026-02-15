# config.py - Configuration settings
from pathlib import Path

class Config:
    # Image settings
    IMG_SIZE = 128
    PIXEL_TO_MM = 0.7  # Standard CT conversion
    
    # Paths
    BASE_DIR = Path(".")
    SAMPLE_IMAGES_DIR = BASE_DIR / "sample_images"
    MODEL_DIR = BASE_DIR / "saved_models"
    OUTPUT_DIR = BASE_DIR / "outputs"
    
    # Create directories
    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)