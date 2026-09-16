from pathlib import Path
import torch

class Config:
    IMG_SIZE = 256
    CROP_SIZE = 256
    DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    
    YOLO_MODEL_PATH = Path("models/yolo_final.pt")
    UNET_MODEL_PATH = Path("models/surgeon_unet_corrected_best.pth")
    
    YOLO_CONF_THRESH = 0.25
    UNET_CONF_THRESH = 0.50

# Exports for legacy imports
YOLO_MODEL = Config.YOLO_MODEL_PATH
UNET_MODEL = Config.UNET_MODEL_PATH
DEVICE = Config.DEVICE
DATA_DIR = Path("datasets/kits19")