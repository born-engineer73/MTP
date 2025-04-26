import os
import sys
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

sys.path.append('D:/Vaibhav Hacker/Desktop/MTP/pytorch-i3d')
from pytorch_i3d import InceptionI3d

# ========== PARAMETERS ==========
VIDEO_DIR = 'D:/Vaibhav Hacker/Desktop/MTP/videos/testing2/'
OUTPUT_DIR = 'D:/Vaibhav Hacker/Desktop/MTP/Dataset/testing2/features_rgb/'
MODEL_PATH = 'D:/Vaibhav Hacker/Desktop/MTP/pytorch-i3d/models/rgb_imagenet.pt'
FPS = 30
NUM_FRAMES = 64
IMAGE_SIZE = 224  # I3D input resolution

# ========== TRANSFORMS ==========
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# ========== LOAD I3D MODEL ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

i3d = InceptionI3d(400, in_channels=3)
i3d.load_state_dict(torch.load(MODEL_PATH))
i3d.to(device)
i3d.eval()

# ========== FUNCTION ==========
def video_to_npy(video_path, out_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = transforms.ToPILImage()(frame)
        frame = transform(pil)
        frames.append(frame)
    
    cap.release()
    frames = torch.stack(frames)  # shape: (T, C, H, W)
    frames = frames.permute(1, 0, 2, 3)  # (C, T, H, W)

    with torch.no_grad():
        features = i3d.extract_features(frames.unsqueeze(0).to(device))  # (1, 1024, T//8)
        features = features.squeeze().permute(1, 0).cpu().numpy()        # (T//8, 1024)

    np.save(out_path, features)
    print(f"Saved: {out_path}")

# ========== MAIN LOOP ==========
os.makedirs(OUTPUT_DIR, exist_ok=True)
video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(('.mp4', '.avi', '.mov'))]

for vid in video_files:
    vid_path = os.path.join(VIDEO_DIR, vid)
    out_path = os.path.join(OUTPUT_DIR, vid.replace('.mp4', '.npy').replace('.avi', '.npy'))
    video_to_npy(vid_path, out_path)

