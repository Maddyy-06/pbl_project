import torch
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from model import SurgeonUNet
from preprocess import process_nifti, get_crop
from config import YOLO_MODEL, UNET_MODEL, DEVICE

def run_inference(case_path, slice_idx=None):
    scout = YOLO(YOLO_MODEL)
    surgeon = SurgeonUNet().to(DEVICE)
    surgeon.load_state_dict(torch.load(UNET_MODEL))
    surgeon.eval()

    img_vol = process_nifti(f"{case_path}/imaging.nii")
    
    if slice_idx is None:
        slice_idx = img_vol.shape[0] // 2
        
    raw = img_vol[slice_idx]
    img_3ch = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    
    res = scout(img_3ch, verbose=False)[0]
    if len(res.boxes) > 0:
        box = res.boxes.xyxy[0].cpu().numpy().astype(int)
        crop = get_crop(raw, box)
        
        tensor = torch.tensor(crop).float().view(1, 1, 128, 128).to(DEVICE) / 255.0
        with torch.no_grad():
            mask = (surgeon(tensor).cpu().numpy()[0, 0] > 0.5).astype(np.uint8)

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(raw, cmap='gray')
        ax[1].imshow(raw, cmap='gray')
        ax[1].add_patch(plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], fill=False, color='red', lw=2))
        ax[2].imshow(crop, cmap='gray')
        ax[2].imshow(mask, cmap='Reds', alpha=0.4)
        plt.show()

if __name__ == "__main__":
    # Change this to your local data path
    run_inference("./datasets/kits19/case_00255")