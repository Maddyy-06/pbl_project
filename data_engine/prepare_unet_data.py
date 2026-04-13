import numpy as np
import nibabel as nib
from pathlib import Path
import cv2
from tqdm import tqdm

KITS19_ROOT = Path("/Users/madhavangupta/Downloads/kits19/data")
PNG_DIR = Path("sample_images")
OUTPUT_DIR = Path("unet_dataset")

OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "images").mkdir(exist_ok=True)
(OUTPUT_DIR / "masks").mkdir(exist_ok=True)

for seg_path in tqdm(list(KITS19_ROOT.glob("case_*/segmentation.nii.gz"))):
    case_id = seg_path.parent.name
    seg = nib.load(seg_path).get_fdata()
    for slice_idx in range(seg.shape[2]):
        slice_seg = seg[:, :, slice_idx]
        kidney_mask = (slice_seg == 1).astype(np.uint8)
        if kidney_mask.sum() == 0:
            continue

        contours, _ = cv2.findContours(kidney_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        kidney_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(kidney_contour)

        # Add a small margin around the kidney
        margin = 20
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(seg.shape[1] - x, w + 2*margin)
        h = min(seg.shape[0] - y, h + 2*margin)

        img_path = PNG_DIR / case_id / f"slice_{slice_idx:04d}.png"
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        img_crop = img[y:y+h, x:x+w]

        tumor_mask = (slice_seg == 2).astype(np.uint8) * 255
        mask_crop = tumor_mask[y:y+h, x:x+w]

        out_name = f"{case_id}_{slice_idx:04d}.png"
        cv2.imwrite(str(OUTPUT_DIR / "images" / out_name), img_crop)
        cv2.imwrite(str(OUTPUT_DIR / "masks" / out_name), mask_crop)

print("✅ U‑Net dataset ready in 'unet_dataset/'")