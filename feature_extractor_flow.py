
import os
import sys
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F

sys.path.append('D:/Vaibhav Hacker/Desktop/MTP/pytorch-i3d')
from pytorch_i3d import InceptionI3d

# ========== PARAMETERS ==========
VIDEO_DIR = 'D:/Vaibhav Hacker/Desktop/MTP/videos/testing2/'
OUTPUT_DIR = 'D:/Vaibhav Hacker/Desktop/MTP/Dataset/testing2/features_flow/'
MODEL_PATH = 'D:/Vaibhav Hacker/Desktop/MTP/pytorch-i3d/models/flow_imagenet.pt'
FPS = 30
IMAGE_SIZE = 224  # I3D input resolution

# ========== LOAD I3D FLOW MODEL ==========
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")

i3d = InceptionI3d(400, in_channels=2)  # Flow has 2 channels
i3d.load_state_dict(torch.load(MODEL_PATH))
i3d.to(device)
i3d.eval()

# ========== FUNCTION ==========
def video_to_flow_npy(video_path, out_path):
    cap = cv2.VideoCapture(video_path)
    prev_gray = None
    flow_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray,
                None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )  # shape: (H, W, 2)

            # Resize and normalize flow
            flow = cv2.resize(flow, (IMAGE_SIZE, IMAGE_SIZE))
            flow_tensor = torch.from_numpy(flow).permute(2, 0, 1).float() / 255.0  # (2, H, W)
            flow_tensor = F.normalize(flow_tensor, mean=[0.5, 0.5], std=[0.5, 0.5])
            flow_frames.append(flow_tensor)

        prev_gray = gray

    cap.release()

    if len(flow_frames) < 1:
        print(f"[WARNING] Skipping {video_path} due to too few frames.")
        return

    frames = torch.stack(flow_frames)  # (T, 2, H, W)
    frames = frames.permute(1, 0, 2, 3)  # (2, T, H, W)

    with torch.no_grad():
        features = i3d.extract_features(frames.unsqueeze(0).to(device))  # (1, 1024, T//8)
        features = features.squeeze().permute(1, 0).cpu().numpy()        # (T//8, 1024)

    np.save(out_path, features)
    print(f" Saved: {out_path}")

# ========== MAIN LOOP ==========
os.makedirs(OUTPUT_DIR, exist_ok=True)
video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(('.mp4', '.avi', '.mov'))]

for vid in video_files:
    vid_path = os.path.join(VIDEO_DIR, vid)
    out_path = os.path.join(OUTPUT_DIR, vid.replace('.mp4', '.npy').replace('.avi', '.npy'))
    video_to_flow_npy(vid_path, out_path)