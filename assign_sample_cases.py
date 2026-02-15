# assign_sample_cases.py - CORRECT SLICE NUMBERS FROM YOUR FILES

import shutil
from pathlib import Path

print("=" * 60)
print("Assigning Verified KiTS19 PNGs to Sample Cases")
print("=" * 60)

# Paths
SAMPLE_IMAGES_DIR = Path("sample_images")
OUTPUT_DIR = Path("sample_cases")
OUTPUT_DIR.mkdir(exist_ok=True)

# ✅ VERIFIED EXACT SLICES FROM YOUR FILES - DO NOT CHANGE
VERIFIED_SLICES = {
    "normal": ("case_00000", 145),      # slice_0145.png - Normal kidney, NO tumor
    "cyst": ("case_00012", 154),        # slice_0154.png - Benign cyst
    "small_tumor": ("case_00005", 264), # slice_0264.png - Small tumor
    "large_tumor": ("case_00056", 88),  # slice_0088.png - Large tumor
    "multiple_tumors": ("case_00102", 134) # slice_0134.png - Multiple tumors
}

print("\n📋 Copying verified PNGs to sample_cases/ (NO RESIZE, NO CROP)...")
print("-" * 60)

# Clear old files
for f in OUTPUT_DIR.glob("*.png"):
    f.unlink()

for category, (case_folder, slice_num) in VERIFIED_SLICES.items():
    source_path = SAMPLE_IMAGES_DIR / case_folder / f"slice_{slice_num:04d}.png"
    
    if source_path.exists():
        # Copy with new name - EXACT COPY, NO MODIFICATION
        clean_name = {
            "normal": "normal_kidney.png",
            "cyst": "benign_cyst.png",
            "small_tumor": "small_tumor.png",
            "large_tumor": "large_tumor.png",
            "multiple_tumors": "multiple_tumors.png"
        }[category]
        
        clean_path = OUTPUT_DIR / clean_name
        shutil.copy2(source_path, clean_path)
        
        # Get file size to verify it's untouched
        file_size = clean_path.stat().st_size / 1024
        print(f"  ✅ {category:15} -> {clean_name:20} ({file_size:.1f} KB) - ORIGINAL SIZE, NO RESIZE")
    else:
        print(f"  ❌ {category:15} -> File not found: {source_path}")

print("\n" + "=" * 60)
print("✅ ASSIGNMENT COMPLETE!")
print("=" * 60)
print("\n📁 Your sample images are ready in 'sample_cases/':")
print("   1. normal_kidney.png      (Normal - NO tumor) - from slice_0145")
print("   2. benign_cyst.png        (Benign cyst) - from slice_0154")
print("   3. small_tumor.png        (Small tumor) - from slice_0264")
print("   4. large_tumor.png        (Large tumor) - from slice_0088")
print("   5. multiple_tumors.png    (Multiple tumors) - from slice_0134")