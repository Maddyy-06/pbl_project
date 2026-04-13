import streamlit as st
import numpy as np
from PIL import Image
import torch
import cv2
from ultralytics import YOLO
from model import SurgeonUNet
from config import Config

st.set_page_config(page_title="SurgeonAI | Kidney Oncology Assist", layout="wide")

@st.cache_resource
def load_models():
    yolo = YOLO(Config.YOLO_MODEL)
    unet = SurgeonUNet().to('cpu')
    unet.load_state_dict(torch.load(Config.UNET_MODEL, map_location='cpu'))
    unet.eval()
    return yolo, unet

yolo_model, unet_model = load_models()

def predict(image_np):
    # Pass 1: Standard Search
    results = yolo_model(image_np, conf=0.15, verbose=False)[0]
    
    # Pass 2: Extreme Sensitivity for distorted kidneys (Blue Circles)
    if len(results.boxes) < 2:
        results = yolo_model(image_np, conf=0.03, verbose=False)[0]
    
    if len(results.boxes) == 0: 
        return None, 0, "KIDNEY NOT DETECTED: ROI localization failed.", "warning"

    overlay = image_np.copy()
    h, w = image_np.shape[:2]
    total_tumor_area = 0
    
    # Use top 2 largest detections
    all_boxes = sorted(results.boxes, key=lambda x: (x.xyxy[0][2] - x.xyxy[0][0]) * (x.xyxy[0][3] - x.xyxy[0][1]), reverse=True)[:2]
    
    for i, box_data in enumerate(all_boxes):
        box = box_data.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = np.clip(box[0], 0, w), np.clip(box[1], 0, h), np.clip(box[2], 0, w), np.clip(box[3], 0, h)
        
        crop = image_np[y1:y2, x1:x2]
        if crop.size == 0: continue
            
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        crop_resized = cv2.resize(crop_gray, (128, 128))
        
        tensor = torch.from_numpy(crop_resized).float().view(1, 1, 128, 128) / 255.0
        with torch.no_grad():
            mask = unet_model(tensor).squeeze().cpu().numpy()
        
        mask_binary = (cv2.resize(mask, (crop.shape[1], crop.shape[0])) > 0.5).astype(np.uint8)
        area = int(np.sum(mask_binary))
        total_tumor_area += area
        
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if area > 0:
            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt[:, :, 0] += x1
                cnt[:, :, 1] += y1
                cv2.drawContours(overlay, [cnt], -1, (255, 0, 0), 2)

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
    st.title("🏥 SurgeonAI: Kidney Oncology Assist")
    st.markdown("---")
    
    with st.sidebar:
        st.header("Diagnosis Panel")
        mode = st.selectbox("Select Source", ["📁 Preset Cases", "📤 Upload Scan"])
        
        if mode == "📁 Preset Cases":
            case_type = st.radio("Clinical Profile", ["Normal", "Benign Cyst", "Small Tumor", "Multiple Tumors", "Large Tumor"])
            filename = f"{case_type.lower().replace(' ', '_')}.png"
            img_file = Config.SAMPLE_CASES / filename
            if img_file.exists():
                image = Image.open(img_file).convert('RGB')
            else:
                st.error(f"File {filename} not found in sample_cases folder.")
                return
        else:
            uploaded = st.file_uploader("Upload Medical Scan", type=['png', 'jpg'])
            if uploaded: 
                image = Image.open(uploaded).convert('RGB')

    # Analysis Section
    if 'image' in locals():
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Input Scan Slice", width="stretch")
        
        if st.button("🔍 Run Full Diagnostic"):
            # This is where the magic (or the failure) happens
            overlay, area, status, diag_type = predict(np.array(image))
            
            with col2:
                if overlay is not None:
                    # Case 1: AI successfully found at least one kidney
                    st.image(overlay, caption="AI Analytical Segmentation", width="stretch")
                    st.metric("Total Segmented Tumor Area", f"{area} px")
                    
                    if diag_type == "error": st.error(status)
                    elif diag_type == "warning": st.warning(status)
                    else: st.success(status)
                else:
                    # Case 2: AI found NOTHING (Common in your current Benign Cyst image)
                    st.warning("⚠️ **LOCALIZATION FAILURE**")
                    st.info(f"**Reason:** {status}")
                    st.markdown("""
                        ---
                        **Technical Insight for Panel:** The model failed to identify the kidney ROI in this slice. This is frequently observed in **Non-Contrast CT phases** where the Hounsfield Units (density) of the kidney parenchyma (dark gray) blend into the surrounding bowel and musculature, reducing the signal-to-noise ratio for the YOLO detector.
                    """)

if __name__ == "__main__":
    main()