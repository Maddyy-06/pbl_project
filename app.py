import streamlit as st
import numpy as np
from PIL import Image
import torch
import cv2
from ultralytics import YOLO
# Ensure these files (model.py and config.py) are in the same directory
from model import SurgeonUNet
from config import Config

# 1. Setup Page & Title
st.set_page_config(page_title="Kidney Tumor Detection", layout="wide")

@st.cache_resource
def load_models():
    # Loading YOLO and U-Net as per your research pipeline
    yolo = YOLO(Config.YOLO_MODEL)
    unet = SurgeonUNet().to('cpu')
    unet.load_state_dict(torch.load(Config.UNET_MODEL, map_location='cpu'))
    unet.eval()
    return yolo, unet

yolo_model, unet_model = load_models()

def predict(image_np):
    """
    Dual-Stage Pipeline: 
    Stage 1: YOLOv8 for Kidney ROI Localization
    Stage 2: U-Net for Tumor Segmentation
    """
    # Pass 1: Standard Search
    results = yolo_model(image_np, conf=0.15, verbose=False)[0]
    
    # Pass 2: Extreme Sensitivity for distorted kidneys
    if len(results.boxes) < 2:
        results = yolo_model(image_np, conf=0.03, verbose=False)[0]
    
    if len(results.boxes) == 0: 
        return None, 0, "KIDNEY NOT DETECTED: ROI localization failed.", "warning"

    overlay = image_np.copy()
    h, w = image_np.shape[:2]
    total_tumor_area = 0
    
    # Take top 2 largest detections (Left and Right Kidney)
    all_boxes = sorted(results.boxes, key=lambda x: (x.xyxy[0][2] - x.xyxy[0][0]) * (x.xyxy[0][3] - x.xyxy[0][1]), reverse=True)[:2]
    
    for i, box_data in enumerate(all_boxes):
        box = box_data.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = np.clip(box[0], 0, w), np.clip(box[1], 0, h), np.clip(box[2], 0, w), np.clip(box[3], 0, h)
        
        crop = image_np[y1:y2, x1:x2]
        if crop.size == 0: continue
            
        # Preprocessing for U-Net
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        crop_resized = cv2.resize(crop_gray, (128, 128))
        
        tensor = torch.from_numpy(crop_resized).float().view(1, 1, 128, 128) / 255.0
        with torch.no_grad():
            mask = unet_model(tensor).squeeze().cpu().numpy()
        
        # Post-processing Mask
        mask_binary = (cv2.resize(mask, (crop.shape[1], crop.shape[0])) > 0.5).astype(np.uint8)
        area = int(np.sum(mask_binary))
        total_tumor_area += area
        
        # Drawing Results
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green Box for Kidney
        if area > 0:
            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt[:, :, 0] += x1
                cnt[:, :, 1] += y1
                cv2.drawContours(overlay, [cnt], -1, (255, 0, 0), 2) # Red Contour for Tumor

    # Diagnostic Logic
    if total_tumor_area > 100:
        status = "MALIGNANT: Significant Tumor Mass Detected"
        diag_type = "error"
    elif total_tumor_area > 20: 
        status = "SUSPICIOUS: Potential Lesion Identified"
        diag_type = "warning"
    else:
        status = "NORMAL: No Clinically Significant Findings"
        diag_type = "success"
        
    return overlay, total_tumor_area, status, diag_type

def main():
    st.title("🔬 Kidney Tumor Detection")
    st.write("Registration ID: 2427030337 | Guide: Dr. Ridhi Arora")
    st.markdown("---")
    
    # Global variable to hold the image to be processed
    input_image = None

    with st.sidebar:
        st.header("Upload & Case Selection")
        mode = st.selectbox("Select Source", ["📁 Preset Cases", "📤 Upload Scan"])
        
        if mode == "📁 Preset Cases":
            case_type = st.radio("Clinical Profile", ["Normal", "Benign Cyst", "Small Tumor", "Multiple Tumors", "Large Tumor"])
            filename = f"{case_type.lower().replace(' ', '_')}.png"
            img_path = Config.SAMPLE_CASES / filename
            if img_path.exists():
                input_image = Image.open(img_path).convert('RGB')
            else:
                st.error(f"Sample {filename} not found.")
        else:
            uploaded = st.file_uploader("Upload Medical Scan from System", type=['png', 'jpg', 'jpeg'])
            if uploaded: 
                input_image = Image.open(uploaded).convert('RGB')

    # If an image exists (either from preset or upload), show the UI
    if input_image is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### 1. Input Scan Slice")
            st.image(input_image, use_container_width=True)
            
            # The Analysis Trigger Button
            analyze_btn = st.button("🔍 RUN FULL DIAGNOSTIC", use_container_width=True)
        
        if analyze_btn:
            with st.spinner("Executing Dual-Stage AI Pipeline..."):
                # Convert PIL to Numpy for Processing
                overlay, area, status, diag_type = predict(np.array(input_image))
            
            with col2:
                st.write("### 2. AI Analytical Output")
                if overlay is not None:
                    st.image(overlay, use_container_width=True, caption="YOLOv8 + U-Net Visualization")
                    st.metric("Total Segmented Tumor Area", f"{area} px")
                    
                    if diag_type == "error": st.error(status)
                    elif diag_type == "warning": st.warning(status)
                    else: st.success(status)
                else:
                    st.warning("⚠️ **LOCALIZATION FAILURE**")
                    st.info(f"**Reason:** {status}")
                    st.markdown("""
                        ---
                        **Technical Note:** The localization failed to detect a kidney ROI. 
                        This usually occurs in non-contrast phases where kidney density 
                        is similar to surrounding tissue.
                    """)
    else:
        st.info("Please select a preset case or upload a scan from your system to begin.")

if __name__ == "__main__":
    main()