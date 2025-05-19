import os
import cv2
from tqdm import tqdm
import numpy as np

def horizontal_flip(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        flipped = cv2.flip(frame, 1)
        out.write(flipped)
    cap.release()
    out.release()

def increase_brightness(input_path, output_path, value=30):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[..., 2] = np.clip(hsv[..., 2] + value, 0, 255)
        bright = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        out.write(bright)
    cap.release()
    out.release()

def apply_augmentations_to_all_videos(video_dir):
    video_files = sorted([f for f in os.listdir(video_dir) if f.endswith(('.avi', '.mp4'))], key=lambda x: int(x.split('.')[0]))
    count = len(video_files) + 1

    print("\n🔁 Step 1: Applying Horizontal Flip to all videos")
    for fname in tqdm(video_files, desc="Horizontal Flip"):
        path = os.path.join(video_dir, fname)
        out_path = os.path.join(video_dir, f"{count}.avi")
        horizontal_flip(path, out_path)
        count += 1



    print("\n🌞 Step 2: Applying Brightness Increase to all original videos")
    for fname in tqdm(video_files, desc="Brightness Increase"):
        path = os.path.join(video_dir, fname)
        out_path = os.path.join(video_dir, f"{count}.avi")
        increase_brightness(path, out_path)
        count += 1

    print(f"\n✅ All augmentations done. Total videos now: {count - 1}")

# Example usage:
video_dir = 'D:/Vaibhav Hacker/Desktop/MTP/videos/Withdraw'
apply_augmentations_to_all_videos(video_dir)
