# app.py - Kidney Tumor Detection System
# USES EXACT PNG COPIES - NO RESIZE, NO CROP

import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path
import datetime
import cv2

st.set_page_config(page_title="Kidney Tumor Detection", page_icon="🏥", layout="wide")

st.markdown("""
<style>
.main-header { font-size: 2.5rem; color: #2c3e50; text-align: center; padding: 1.5rem; font-weight: 600; }
.stButton button { background-color: #3498db; color: white; font-weight: 600; border: none; padding: 0.75rem 1.5rem; border-radius: 8px; width: 100%; }
.result-card { background: #f8f9fa; padding: 1.5rem; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
.footer { text-align: center; margin-top: 3rem; padding: 1rem; color: #95a5a6; border-top: 1px solid #ecf0f1; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'annotated_image' not in st.session_state:
    st.session_state.annotated_image = None

# Path to sample cases - EXACT PNG COPIES
SAMPLE_CASES_DIR = Path("sample_cases")

def generate_synthetic_normal():
    """Generate guaranteed tumor-free kidney image"""
    size = 256
    x = np.linspace(-2, 2, size)
    y = np.linspace(-2, 2, size)
    X, Y = np.meshgrid(x, y)
    img = np.exp(-(X**2 + 0.5*(Y+0.3)**2))
    img += 0.3 * np.exp(-((X-0.7)**2 + (Y-0.4)**2)/0.4)
    img = img / img.max()
    img = img * 0.4 + 0.3
    img = np.clip(img, 0, 1)
    return Image.fromarray((img * 255).astype(np.uint8))

def get_sample_image(case_type):
    """Load EXACT PNG copy - NO RESIZE, NO CROP, NO MODIFICATION"""
    
    # ✅ ALL REAL IMAGES - NO SYNTHETIC, NO BLUR
    verified_files = {
        "Normal": "normal_kidney.png",      # REAL image from case_00000
        "Benign Cyst": "benign_cyst.png",    # REAL image from case_00012
        "Small Tumor": "small_tumor.png",    # REAL image from case_00005
        "Large Tumor": "large_tumor.png",    # REAL image from case_00056
        "Multiple Tumors": "multiple_tumors.png"  # REAL image from case_00102
    }
    
    filename = verified_files.get(case_type)
    if filename:
        img_path = SAMPLE_CASES_DIR / filename
        if img_path.exists():
            # LOAD EXACT COPY - NO RESIZE, NO CROP, NO MODIFICATION
            return Image.open(img_path).convert('RGB')
    
    # Fallback - should never happen
    return Image.fromarray(np.ones((512, 512, 3), dtype=np.uint8) * 200)

def preprocess_image(image):
    """Resize ONLY for model input - display remains ORIGINAL"""
    img = image.resize((128, 128))
    return np.array(img) / 255.0

def detect_abnormalities(img_array):
    try:
        gray = cv2.cvtColor((img_array * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        findings = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 80:
                x, y, w, h = cv2.boundingRect(contour)
                findings.append({'area': area, 'bbox': (x, y, w, h)})
        return findings
    except:
        return []

def calculate_tumor_size_mm(bbox):
    return round(max(bbox[2], bbox[3]) * 0.7, 1)

def assess_risk(size_mm):
    if size_mm == 0:
        return "No Abnormality", "Normal appearance. Routine follow-up recommended."
    elif size_mm < 10:
        return "Low Risk", "Probably benign. Annual follow-up."
    elif size_mm < 40:
        return "Medium Risk", "Indeterminate. 6-month follow-up advised."
    else:
        return "High Risk", "Suspicious for malignancy. Urology consult recommended."

def create_annotated_image(original_img, findings):
    img = original_img.copy()
    try:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        for f in findings:
            x, y, w, h = f['bbox']
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return img
    except:
        return img

def main():
    st.markdown('<h1 class="main-header">🏥 Kidney Tumor Detection</h1>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("### Input Mode")
        mode = st.radio("Select source:", ["📁 Sample Cases", "📤 Upload Image"], index=0)
        
        if st.button("🔄 Clear Session", use_container_width=True):
            st.session_state.uploaded_image = None
            st.session_state.analysis_results = None
            st.session_state.annotated_image = None
            st.rerun()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Input Image")
        
        if mode == "📁 Sample Cases":
            case = st.selectbox(
                "Select case:",
                ["Normal", "Benign Cyst", "Small Tumor", "Large Tumor", "Multiple Tumors"]
            )
            
            image = get_sample_image(case)
            st.session_state.uploaded_image = image
            # DISPLAY ORIGINAL SIZE - NO RESIZE
            st.image(image, caption=f"Sample: {case}", use_column_width=True)
        
        else:
            uploaded_file = st.file_uploader("Choose a CT scan", type=['png', 'jpg', 'jpeg', 'tiff'])
            if uploaded_file:
                image = Image.open(uploaded_file).convert('RGB')
                st.session_state.uploaded_image = image
                st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.session_state.uploaded_image:
            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("Analyzing..."):
                    img = preprocess_image(st.session_state.uploaded_image)
                    findings = detect_abnormalities(img)
                    
                    size = 0
                    if findings:
                        largest = max(findings, key=lambda x: x['area'])
                        size = calculate_tumor_size_mm(largest['bbox'])
                    
                    risk, rec = assess_risk(size)
                    annotated = create_annotated_image((img * 255).astype(np.uint8), findings)
                    
                    st.session_state.analysis_results = {
                        'abnormalities': len(findings),
                        'size': size,
                        'risk': risk,
                        'recommendation': rec
                    }
                    st.session_state.annotated_image = annotated
                    st.success("Analysis complete!")
    
    with col2:
        st.markdown("### Analysis Results")
        
        if st.session_state.analysis_results:
            r = st.session_state.analysis_results
            
            if st.session_state.annotated_image is not None:
                st.image(st.session_state.annotated_image, caption="Detection results", use_column_width=True)
            
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            
            if r['abnormalities'] > 0:
                st.markdown(f"**Finding:** ✅ Detected ({r['abnormalities']} region{'s' if r['abnormalities'] > 1 else ''})")
                st.metric("Size", f"{r['size']} mm")
            else:
                st.markdown("**Finding:** ❌ No abnormality detected")
            
            risk = r['risk']
            emoji = "🔴" if "High" in risk else "🟠" if "Medium" in risk else "🟢" if "Low" in risk else "⚪"
            st.markdown(f"**Risk Level:** {emoji} {risk}")
            st.markdown(f"**Recommendation:** {r['recommendation']}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            report = f"""KIDNEY CT ANALYSIS REPORT
Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

FINDINGS:
- Abnormality: {'Detected' if r['abnormalities'] > 0 else 'None'}
- Size: {r['size']} mm
- Risk Level: {r['risk']}

RECOMMENDATION:
{r['recommendation']}

---
This is an AI-assisted analysis tool. All findings should be verified by a medical professional.
"""
            st.download_button(
                label="📄 Download Report",
                data=report,
                file_name=f"kidney_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        else:
            st.info("👈 Select a sample case or upload an image to begin analysis")
    
    st.markdown("---")
    st.markdown('<div class="footer">Kidney Tumor Detection System | Phase 1: Computer Vision</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()