import cv2
import os
import numpy as np
from tqdm import tqdm
from multiprocessing import Process
import shutil
import torch
import torchvision.transforms as transforms
import sys
import torchvision.transforms.functional as F
import subprocess
from multiprocessing import Process, Manager
scripts = [
    "D:/Vaibhav Hacker/Desktop/MTP/label.py",
    "D:/Vaibhav Hacker/Desktop/MTP/Final_MSTCN_repo/final_main.py"
]


sys.path.append('D:/Vaibhav Hacker/Desktop/MTP/pytorch-i3d')
from pytorch_i3d import InceptionI3d

# --- CONFIGURATION ---
# file_name='test.npy'
FRAMES_FOLDER = "D:/Vaibhav Hacker/Desktop/MTP/frames/"
RGB_FEATURE_FOLDER = "D:/Vaibhav Hacker/Desktop/MTP/rgb_features"

FLOW_FEATURE_FOLDER = "D:/Vaibhav Hacker/Desktop/MTP/flow_features"

COMBINED_FEATURES_FOLDER = "D:/Vaibhav Hacker/Desktop/MTP/combined_features"
FOLDERS= [ RGB_FEATURE_FOLDER, FLOW_FEATURE_FOLDER, COMBINED_FEATURES_FOLDER]

npy_file_paths = {}


# === Step 3: Create and save .npy files ===
# Dummy features array (you can change this per folder if needed)
# features = np.random.rand(10, 224, 224, 3)

# for folder, npy_path in npy_file_paths.items():
#     np.save(npy_path, features)
#     print(f"Saved .npy file at: {npy_path}")
# --- CLEANUP ---
def clean_folders(npy_file_paths):
    
    for folder in FOLDERS:
        npy_file_paths[folder] = os.path.join(folder, 'test.npy')
    for folder in FOLDERS:
        if os.path.exists(folder):
            print("Emptying")
            shutil.rmtree(folder)  # Delete folder if exists
        os.makedirs(folder)         # Recreate fresh folder
        print(f"Directory prepared: {folder}")
    
    print(npy_file_paths)
    if os.path.exists(FRAMES_FOLDER):
        print("Emptying")
        shutil.rmtree(FRAMES_FOLDER)  # Delete folder if exists
    os.makedirs(FRAMES_FOLDER)         # Recreate fresh folder
    print(f"Directory prepared: {FRAMES_FOLDER}")
    
# --- STEP 1: Capture Video and Save Frames ---
def capture_video(output_path='output.avi', fps=30.0):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot access webcam.")
        return
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(output_path,
                          cv2.VideoWriter_fourcc(*'XVID'),
                          fps,
                          (frame_width, frame_height))
    frame_count = 0
    print("Recording... Press 'q' to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_path = os.path.join(FRAMES_FOLDER, f"frame_{frame_count:05d}.jpg")
        cv2.imwrite(frame_path, frame)

        cv2.imshow('Recording', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {frame_count} frames.")

# --- STEP 2: Extract RGB Features ---
def extract_rgb_features(npy_file_paths):
    print(npy_file_paths)
    MODEL_PATH = 'D:/Vaibhav Hacker/Desktop/MTP/pytorch-i3d/models/rgb_imagenet.pt'
    IMAGE_SIZE = 224 
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
    frame_files = sorted(os.listdir(FRAMES_FOLDER))
    frames = []
    for frame in tqdm(frame_files, desc="Extracting RGB Features"):
    # for frame in frame_files:
        # print(frame)
        frame= os.path.join(FRAMES_FOLDER, frame)
        frame=cv2.imread(frame)
        # frame_path = os.path.join(FRAMES_FOLDER, frame_file)
        # img = cv2.imread(frame_path)

        # # Simple RGB feature extraction: flatten and normalize
        # features = img.flatten() / 255.0
        
        # # Save features
        # np.save(os.path.join(RGB_FEATURE_FOLDER, frame_file.replace(".jpg", ".npy")), features)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = transforms.ToPILImage()(frame)
        frame = transform(pil)
        frames.append(frame)
    frames = torch.stack(frames)  # shape: (T, C, H, W)
    frames = frames.permute(1, 0, 2, 3)  # (C, T, H, W)

    with torch.no_grad():
        features = i3d.extract_features(frames.unsqueeze(0).to(device))  # (1, 1024, T//8)
        features = features.squeeze().permute(1, 0).cpu().numpy()        # (T//8, 1024)
    print(npy_file_paths)
    np.save(npy_file_paths[RGB_FEATURE_FOLDER], features)
    print(f"Saved: {RGB_FEATURE_FOLDER}")
# --- STEP 3: Extract Optical Flow Features ---
def extract_flow_features(npy_file_paths):
    frame_files = sorted(os.listdir(FRAMES_FOLDER))
    prev_gray = None
    MODEL_PATH = 'D:/Vaibhav Hacker/Desktop/MTP/pytorch-i3d/models/flow_imagenet.pt'
    FPS = 30
    IMAGE_SIZE = 224
    device = torch.device("cuda")

    i3d = InceptionI3d(400, in_channels=2)  # Flow has 2 channels
    i3d.load_state_dict(torch.load(MODEL_PATH))
    i3d.to(device)
    i3d.eval()

    prev_gray = None
    flow_frames = []

    for frame in tqdm(frame_files, desc="Extracting Flow Features"):
    # for frame in frame_files:
        # print(frame)
        frame= os.path.join(FRAMES_FOLDER, frame)
        frame=cv2.imread(frame)
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

    if len(flow_frames) < 1:
        print(f"[WARNING] Skipping {FRAMES_FOLDER} due to too few frames.")
        return

    frames = torch.stack(flow_frames)  # (T, 2, H, W)
    frames = frames.permute(1, 0, 2, 3)  # (2, T, H, W)

    with torch.no_grad():
        features = i3d.extract_features(frames.unsqueeze(0).to(device))  # (1, 1024, T//8)
        features = features.squeeze().permute(1, 0).cpu().numpy()        # (T//8, 1024)

    np.save(npy_file_paths[FLOW_FEATURE_FOLDER], features)
    print(f" Saved: {FLOW_FEATURE_FOLDER}")

# --- STEP 4: Combine RGB and Flow Features ---
def combine_features(npy_file_paths):
    rgb_feat = np.load(npy_file_paths[RGB_FEATURE_FOLDER])   # shape: (T, 1024)
    flow_feat = np.load(npy_file_paths[FLOW_FEATURE_FOLDER]) # shape: (T, 1024)

    if rgb_feat.shape != flow_feat.shape:
        last_flow = flow_feat[-1:]
        flow_feat = np.concatenate([flow_feat, last_flow], axis=0)

        # raise ValueError(f"Shape mismatch: RGB {rgb_feat.shape}, Flow {flow_feat.shape}")

    combined = np.concatenate((rgb_feat, flow_feat), axis=1)  # shape: (T, 2048)
    combined = combined.T
    np.save(npy_file_paths[COMBINED_FEATURES_FOLDER], combined)
    print(f"Combined feature saved to: {COMBINED_FEATURES_FOLDER}")
    print(np.shape(combined))

def extract_pattern(txt_file_path):
    with open(txt_file_path, 'r') as f:
        labels = f.read().splitlines()

    pattern = []
    previous_label = None

    for label in labels:
        if label != previous_label:
            pattern.append(label)
            previous_label = label

    # Print the pattern nicely
    print(" -> ".join(pattern))

    # Return the pattern as a list
    return pattern[1:]

# --- MAIN PIPELINE ---
if __name__ == "__main__":
    manager = Manager()
    npy_file_paths = manager.dict()  

    clean_folders(npy_file_paths)    # Pass npy_file_paths to the function

    capture_video()

    p1 = Process(target=extract_rgb_features, args=(npy_file_paths,))
    p2 = Process(target=extract_flow_features, args=(npy_file_paths,))

    p1.start()
    p2.start()

    p1.join()
    p2.join()

    combine_features(npy_file_paths)

    print("\n✅ All steps completed successfully! Features saved in 'combined_features' folder.")
    for script in scripts:
        print(f" Running {script}...")
        result = subprocess.run(["python", script], capture_output=True, text=True)
        
        print(f"Output of {script}:\n{result.stdout}")
        
        if result.stderr:
            print(f"Error in {script}:\n{result.stderr}")
        print("="*50)
    
    txt_file_path = "D:/Vaibhav Hacker/Desktop/MTP/results/test"
    subtask_list = extract_pattern(txt_file_path)

    print("\nSubtask list:", subtask_list)