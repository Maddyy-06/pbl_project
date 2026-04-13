import numpy as np
import nibabel as nib
from pathlib import Path
import cv2

# Adjust these paths
KITS19_ROOT = Path("/Users/madhavangupta/Downloads/kits19/data")   # your KiTS19 folder
PNG_DIR = Path("sample_images")                                     # your PNG slices
OUTPUT_DIR = Path("yolo_dataset")

OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "images").mkdir(exist_ok=True)
(OUTPUT_DIR / "labels").mkdir(exist_ok=True)

for seg_path in KITS19_ROOT.glob("case_*/segmentation.nii.gz"):
    case_id = seg_path.parent.name
    seg = nib.load(seg_path).get_fdata()
    for slice_idx in range(seg.shape[2]):
        slice_seg = seg[:, :, slice_idx]
        # Kidney mask (label = 1)
        kidney_mask = (slice_seg == 1).astype(np.uint8)
        if kidney_mask.sum() == 0:
            continue
        # Find largest contour (the kidney)
        contours, _ = cv2.findContours(kidney_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        kidney_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(kidney_contour)

        img_path = PNG_DIR / case_id / f"slice_{slice_idx:04d}.png"
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        h_img, w_img = img.shape[:2]

        # YOLO format: class_id x_center y_center width height (normalized)
        x_center = (x + w/2) / w_img
        y_center = (y + h/2) / h_img
        width = w / w_img
        height = h / h_img

        label_name = f"{case_id}_{slice_idx:04d}.txt"
        with open(OUTPUT_DIR / "labels" / label_name, 'w') as f:
            f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        # Copy image
        img_dst = OUTPUT_DIR / "images" / f"{case_id}_{slice_idx:04d}.png"
        cv2.imwrite(str(img_dst), img)

print("✅ YOLO dataset ready in 'yolo_dataset/'")