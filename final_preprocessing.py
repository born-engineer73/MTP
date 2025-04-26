import os
import cv2
import numpy as np

VIDEO_DIR = 'D:/Vaibhav Hacker/Desktop/MTP/videos/pick_pour/'
FEATURE_DIR = 'D:/Vaibhav Hacker/Desktop/MTP/Dataset/pick_pour/features_combined/'
LABEL_DIR = 'D:/Vaibhav Hacker/Desktop/MTP/Dataset/pick_pour/ground_truth/'

video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(('.mp4', '.avi', '.mov'))]

print(f"{'Video Name':30} | {'#Frames':>8} | {'Feature Shape':>15} | {'#Label Lines':>13} | {'Status':>25}")
print("-" * 110)

for vid in video_files:
    vid_path = os.path.join(VIDEO_DIR, vid)

    # Count frames in video
    cap = cv2.VideoCapture(vid_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Generate paths
    base_name = os.path.splitext(vid)[0]
    try:
        txt_name = f"{int(base_name)}.txt"
    except ValueError:
        txt_name = f"{base_name}.txt"

    npy_path = os.path.join(FEATURE_DIR, f"{base_name}.npy")
    txt_path = os.path.join(LABEL_DIR, txt_name)

    # Load feature vectors
    if os.path.exists(npy_path):
        features = np.load(npy_path)
        feat_shape = features.shape
        feat_len = feat_shape[1]
    else:
        feat_shape = "Not Found"
        feat_len = None

    # Load label lines
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        label_lines = len(lines)
    else:
        label_lines = "Missing"
        lines = []

    # Adjust label file if mismatch
    status = "✅"
    if isinstance(feat_len, int) and isinstance(label_lines, int):
        if label_lines > feat_len:
            # Trim excess
            with open(txt_path, 'w') as f:
                f.writelines(lines[:feat_len])
            status = f"✂️ Trimmed ({label_lines}->{feat_len})"
            label_lines = feat_len
        elif label_lines < feat_len:
            # Extend by repeating last line
            if lines:  # Make sure file isn't empty
                last_line = lines[-1].strip()  # Remove any trailing newline
                new_lines = [last_line + '\n'] * (feat_len - label_lines)
                updated_lines = [line.rstrip('\n') + '\n' for line in lines] + new_lines

                # Remove the final newline from the last line
                if updated_lines:
                    updated_lines[-1] = updated_lines[-1].rstrip('\n')

                with open(txt_path, 'w') as f:
                    f.writelines(updated_lines)

                status = f"➕ Extended ({label_lines}->{feat_len})"
                label_lines = feat_len

            else:
                status = "⚠️ Empty label file"
    elif feat_len is None:
        status = "⚠️ Missing feature"
    elif label_lines == "Missing":
        status = "⚠️ Missing label"

    print(f"{vid:30} | {num_frames:>8} | {str(feat_shape):>15} | {str(label_lines):>13} | {status:>25}")
