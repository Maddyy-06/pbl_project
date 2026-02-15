# convert_all_kits19_to_png.py - CONVERT ALL KITS19 IMAGES TO PNG
# Run this ONCE to convert your entire KiTS19 dataset

import numpy as np
import nibabel as nib
from pathlib import Path
from PIL import Image
import os

print("=" * 60)
print("KiTS19 to PNG Converter - ALL CASES")
print("=" * 60)

# Path to your KiTS19 dataset
KITS19_DIR = Path.home() / "Downloads/kits19/data"
OUTPUT_DIR = Path("sample_images")
OUTPUT_DIR.mkdir(exist_ok=True)

# Counters
total_cases = 0
total_slices = 0

print(f"\n📁 Reading cases from: {KITS19_DIR}")
print("-" * 60)

# Loop through ALL case folders
for case_folder in sorted(KITS19_DIR.glob("case_*")):
    case_id = case_folder.name.replace("case_", "")
    img_path = case_folder / "imaging.nii.gz"
    
    if img_path.exists():
        try:
            # Load the 3D scan
            img = nib.load(str(img_path)).get_fdata()
            
            # Get dimensions
            depth = img.shape[2]
            total_cases += 1
            
            print(f"\n📊 Case {case_id}: {depth} slices")
            
            # Create case folder in output
            case_output_dir = OUTPUT_DIR / f"case_{case_id}"
            case_output_dir.mkdir(exist_ok=True)
            
            # Convert EVERY slice to PNG
            for slice_idx in range(depth):
                slice_data = img[:, :, slice_idx]
                
                # Skip if slice is mostly empty (background)
                if np.percentile(slice_data, 95) < 10:
                    continue
                
                # Normalize to 0-255
                slice_min = slice_data.min()
                slice_max = slice_data.max()
                if slice_max > slice_min:
                    slice_norm = (slice_data - slice_min) / (slice_max - slice_min)
                    slice_8bit = (slice_norm * 255).astype(np.uint8)
                else:
                    slice_8bit = np.zeros_like(slice_data, dtype=np.uint8)
                
                # Convert grayscale to RGB (3 channels)
                slice_rgb = np.stack([slice_8bit] * 3, axis=-1)
                
                # Save as PNG
                filename = f"slice_{slice_idx:04d}.png"
                output_path = case_output_dir / filename
                Image.fromarray(slice_rgb).save(output_path)
                
                total_slices += 1
                
                # Progress indicator
                if slice_idx % 20 == 0:
                    print(f"    Saved slice {slice_idx}/{depth}")
            
            print(f"  ✅ Case {case_id} complete - {depth} slices")
            
        except Exception as e:
            print(f"  ❌ Error with case {case_id}: {str(e)}")
    else:
        print(f"  ⚠️ Case {case_id}: imaging.nii.gz not found")

print("\n" + "=" * 60)
print("✅ CONVERSION COMPLETE!")
print("=" * 60)
print(f"\n📊 SUMMARY:")
print(f"   - Total cases processed: {total_cases}")
print(f"   - Total slices saved: {total_slices}")
print(f"   - Output directory: {OUTPUT_DIR}")
print(f"\n📁 Folder structure:")
print(f"   sample_images/")
print(f"   ├── case_00000/")
print(f"   │   ├── slice_0000.png")
print(f"   │   ├── slice_0001.png")
print(f"   │   └── ...")
print(f"   ├── case_00001/")
print(f"   └── ...")