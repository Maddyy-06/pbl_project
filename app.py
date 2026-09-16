import streamlit as st
import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO

from model import SurgeonUNet


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

YOLO_PATH = BASE_DIR / "models" / "yolo_final.pt"
UNET_PATH = BASE_DIR / "models" / "surgeon_unet_corrected_best.pth"

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

YOLO_IMGSZ = 512
YOLO_CONF = 0.15

ROI_PADDING = 0.15

UNET_SIZE = 128
UNET_THRESHOLD = 0.50


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Kidney Tumour Detector",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Kidney Tumour Detection & Segmentation")
st.caption("YOLOv8 localization + U-Net segmentation + Grad-CAM")


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    yolo = YOLO(str(YOLO_PATH))

    unet = SurgeonUNet().to(DEVICE)

    checkpoint = torch.load(
        UNET_PATH,
        map_location=DEVICE
    )

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    unet.load_state_dict(
        checkpoint,
        strict=True
    )

    unet.eval()

    return yolo, unet


yolo, unet = load_models()


# ============================================================
# FUNCTIONS
# ============================================================

def detect_tumour(gray):

    rgb = cv2.cvtColor(
        gray,
        cv2.COLOR_GRAY2RGB
    )

    results = yolo.predict(
        source=rgb,
        imgsz=YOLO_IMGSZ,
        conf=YOLO_CONF,
        verbose=False
    )

    if not results:
        return None

    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return None

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()

    best = int(np.argmax(confs))

    x1, y1, x2, y2 = xyxy[best]

    h, w = gray.shape

    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(x1 + 1, min(w, int(x2)))
    y2 = max(y1 + 1, min(h, int(y2)))

    return (
        x1,
        y1,
        x2,
        y2,
        float(confs[best])
    )


def get_padded_roi(gray, box):

    x1, y1, x2, y2 = box

    h, w = gray.shape

    bw = x2 - x1
    bh = y2 - y1

    px = int(bw * ROI_PADDING)
    py = int(bh * ROI_PADDING)

    rx1 = max(0, x1 - px)
    ry1 = max(0, y1 - py)

    rx2 = min(w, x2 + px)
    ry2 = min(h, y2 + py)

    return gray[ry1:ry2, rx1:rx2], (
        rx1, ry1, rx2, ry2
    )


def run_unet(roi):

    resized = cv2.resize(
        roi,
        (UNET_SIZE, UNET_SIZE),
        interpolation=cv2.INTER_LINEAR
    )

    tensor = (
        torch.from_numpy(
            resized.astype(np.float32) / 255.0
        )
        .unsqueeze(0)
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.no_grad():

        logits = unet(tensor)

        probability = torch.sigmoid(
            logits
        )[0, 0].cpu().numpy()

    mask_small = (
        probability >= UNET_THRESHOLD
    ).astype(np.uint8) * 255

    mask = cv2.resize(
        mask_small,
        (roi.shape[1], roi.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    return probability, mask


def gradcam(roi):

    resized = cv2.resize(
        roi,
        (UNET_SIZE, UNET_SIZE),
        interpolation=cv2.INTER_LINEAR
    )

    tensor = (
        torch.from_numpy(
            resized.astype(np.float32) / 255.0
        )
        .unsqueeze(0)
        .unsqueeze(0)
        .to(DEVICE)
    )

    activations = []
    gradients = []

    def forward_hook(module, inp, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    h1 = unet.enc2.register_forward_hook(
        forward_hook
    )

    h2 = unet.enc2.register_full_backward_hook(
        backward_hook
    )

    unet.zero_grad(set_to_none=True)

    logits = unet(tensor)

    probabilities = torch.sigmoid(logits)

    target_mask = probabilities > UNET_THRESHOLD

    if target_mask.any():
        target = (
            logits * target_mask.float()
        ).sum()
    else:
        target = logits.mean()

    target.backward()

    h1.remove()
    h2.remove()

    if not activations or not gradients:
        return np.zeros(
            roi.shape,
            dtype=np.float32
        )

    activation = activations[0]
    gradient = gradients[0]

    weights = gradient.mean(
        dim=(2, 3),
        keepdim=True
    )

    cam = (
        weights * activation
    ).sum(
        dim=1,
        keepdim=True
    )

    cam = torch.relu(cam)

    cam = torch.nn.functional.interpolate(
        cam,
        size=(UNET_SIZE, UNET_SIZE),
        mode="bilinear",
        align_corners=False
    )

    cam = cam[0, 0].detach().cpu().numpy()

    cam_min = cam.min()
    cam_max = cam.max()

    if cam_max > cam_min:
        cam = (
            cam - cam_min
        ) / (
            cam_max - cam_min
        )
    else:
        cam = np.zeros_like(cam)

    return cv2.resize(
        cam,
        (roi.shape[1], roi.shape[0]),
        interpolation=cv2.INTER_LINEAR
    )


# ============================================================
# UPLOAD
# ============================================================

uploaded = st.file_uploader(
    "Upload a CT slice",
    type=["png", "jpg", "jpeg"]
)


if uploaded is not None:

    file_bytes = np.asarray(
        bytearray(uploaded.read()),
        dtype=np.uint8
    )

    gray = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_GRAYSCALE
    )

    if gray is None:
        st.error("Could not read the image.")
        st.stop()

    detection = detect_tumour(gray)

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Original CT")

        st.image(
            gray,
            clamp=True,
            use_container_width=True
        )

    if detection is None:

        st.warning(
            "No tumour detected by YOLOv8."
        )

        st.info(
            "Try another CT slice."
        )

    else:

        x1, y1, x2, y2, confidence = detection

        roi, roi_box = get_padded_roi(
            gray,
            (x1, y1, x2, y2)
        )

        probability, mask = run_unet(
            roi
        )

        rx1, ry1, rx2, ry2 = roi_box

        full_mask = np.zeros_like(gray)

        full_mask[
            ry1:ry2,
            rx1:rx2
        ] = mask

        segmentation_overlay = cv2.cvtColor(
            gray,
            cv2.COLOR_GRAY2BGR
        )

        tumour = full_mask > 0

        segmentation_overlay[tumour] = (
            0.45
            * segmentation_overlay[tumour]
            + 0.55
            * np.array([0, 0, 255])
        ).astype(np.uint8)

        cv2.rectangle(
            segmentation_overlay,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        with col2:

            st.subheader(
                "Tumour Detection + Segmentation"
            )

            st.image(
                segmentation_overlay,
                channels="BGR",
                use_container_width=True
            )

        st.success(
            f"Tumour detected — YOLO confidence: "
            f"{confidence:.3f}"
        )

        tumour_area = int(
            np.count_nonzero(full_mask)
        )

        st.metric(
            "Predicted tumour pixels",
            tumour_area
        )

        st.subheader("Grad-CAM Explainability")

        try:

            cam = gradcam(roi)

            cam_uint8 = (
                np.clip(cam, 0, 1) * 255
            ).astype(np.uint8)

            heatmap = cv2.applyColorMap(
                cam_uint8,
                cv2.COLORMAP_JET
            )

            cam_overlay = cv2.addWeighted(
                cv2.cvtColor(
                    roi,
                    cv2.COLOR_GRAY2BGR
                ),
                0.55,
                heatmap,
                0.45,
                0
            )

            st.image(
                cam_overlay,
                channels="BGR",
                caption="Grad-CAM attention map",
                use_container_width=True
            )

        except Exception as exc:

            st.warning(
                f"Grad-CAM unavailable: {exc}"
            )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Model Information")

    st.write(
        "**Detector:** YOLOv8n"
    )

    st.write(
        "**Segmenter:** SurgeonUNet"
    )

    st.write(
        "**Explainability:** Grad-CAM"
    )

    st.write(
        f"**Device:** {DEVICE}"
    )

    st.divider()

    st.write(
        "Pipeline:"
    )

    st.write(
        "CT → YOLOv8 → padded ROI → "
        "U-Net → tumour mask → Grad-CAM"
    )