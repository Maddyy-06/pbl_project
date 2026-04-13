import numpy as np
import cv2
import nibabel as nib
from config import Config

def process_nifti(path):
    """
    Load a NIfTI volume and apply clinical HU windowing.
    Used for local testing of full volumes.
    """
    vol = nib.load(path).get_fdata()
    # Apply the Kaggle-verified windowing
    vol = np.clip(vol, Config.HU_MIN, Config.HU_MAX)
    # Min-Max Normalization to 0-255
    vol = (vol - vol.min()) / (vol.max() - vol.min()) * 255
    return vol.astype(np.uint8)

def get_crop(img, box):
    """
    Extracts the Kidney ROI from a slice based on YOLO bounding box.
    """
    x1, y1, x2, y2 = box
    crop = img[y1:y2, x1:x2]
    return cv2.resize(crop, Config.CROP_SIZE)