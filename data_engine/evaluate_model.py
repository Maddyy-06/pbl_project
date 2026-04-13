import torch
import numpy as np
import cv2
import nibabel as nib
from model import SurgeonUNet
from preprocess import process_nifti, get_crop
from config import YOLO_MODEL, UNET_MODEL, DEVICE, DATA_DIR
from ultralytics import YOLO

def dice_coeff(pred, target):
    smooth = 1.0
    m1 = pred.flatten()
    m2 = target.flatten()
    intersection = (m1 * m2).sum()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

def evaluate(test_cases):
    scout = YOLO(YOLO_MODEL)
    surgeon = SurgeonUNet().to(DEVICE)
    surgeon.load_state_dict(torch.load(UNET_MODEL))
    surgeon.eval()
    
    scores = []
    for case in test_cases:
        try:
            img_vol = process_nifti(f"{DATA_DIR}/{case}/imaging.nii")
            seg_vol = nib.load(f"{DATA_DIR}/{case}/segmentation.nii").get_fdata()
            
            for i in range(0, img_vol.shape[0], 10):
                if (seg_vol[i] == 2).sum() > 100:
                    raw = img_vol[i]
                    res = scout(cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR), verbose=False)[0]
                    if len(res.boxes) > 0:
                        box = res.boxes.xyxy[0].cpu().numpy().astype(int)
                        crop = get_crop(raw, box)
                        gt_crop = cv2.resize((seg_vol[i][box[1]:box[3], box[0]:box[2]] == 2).astype(float), (128, 128))
                        
                        tensor = torch.tensor(crop).float().view(1, 1, 128, 128).to(DEVICE) / 255.0
                        with torch.no_grad():
                            pred = (surgeon(tensor).cpu().numpy()[0, 0] > 0.5).astype(float)
                        
                        scores.append(dice_coeff(pred, gt_crop))
        except:
            continue
    
    print(f"Mean Dice Score: {np.mean(scores):.4f}")

if __name__ == "__main__":
    test_list = ["case_00150", "case_00151", "case_00152"]
    evaluate(test_list)