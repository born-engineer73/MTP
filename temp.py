# import os
# import cv2
# import numpy as np

# VIDEO_DIR = 'D:/Vaibhav Hacker/Desktop/MTP/videos/pick_place/'
# FEATURE_DIR = 'D:/Vaibhav Hacker/Desktop/MTP/Dataset/pick_place/features_combined/'
# LABEL_DIR = 'D:/Vaibhav Hacker/Desktop/MTP/Dataset/pick_place/ground_truth/'

# video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(('.mp4', '.avi', '.mov'))]

# print(f"{'Video Name':30} | {'#Frames':>8} | {'Feature Shape':>15} | {'#Label Lines':>13} | {'Status':>10}")
# print("-" * 100)

# for vid in video_files:
#     vid_path = os.path.join(VIDEO_DIR, vid)

#     # Count video frames
#     cap = cv2.VideoCapture(vid_path)
#     num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     cap.release()

#     # Feature file
#     base_name = os.path.splitext(vid)[0]
#     npy_path = os.path.join(FEATURE_DIR, f"{base_name}.npy")

#     # Ground truth label file: pad video index to 3 digits
#     try:
#         vid_number = int(base_name)
#         txt_name = f"{vid_number:03d}.txt"
#     except ValueError:
#         txt_name = f"{base_name}.txt"

#     txt_path = os.path.join(LABEL_DIR, txt_name)

#     # Load feature shape
#     if os.path.exists(npy_path):
#         features = np.load(npy_path)
#         feat_shape = features.shape
#     else:
#         feat_shape = "Not Found"

#     # Load label line count
#     if os.path.exists(txt_path):
#         with open(txt_path, 'r') as f:
#             label_lines = sum(1 for _ in f)
#     else:
#         label_lines = "Missing"

#     # Check for mismatch
#     if isinstance(feat_shape, tuple) and isinstance(label_lines, int):
#         status = "✅"
#         if feat_shape[0] != label_lines:
#             status = "⚠️ Mismatch"
#     else:
#         status = "⚠️ Missing"

#     print(f"{vid:30} | {num_frames:>8} | {str(feat_shape):>15} | {str(label_lines):>13} | {status:>10}")


# import os
# import cv2
# import numpy as np

# VIDEO_DIR = 'D:/Vaibhav Hacker/Desktop/MTP/testing_video/'
# FEATURE_DIR_RGB = 'D:/Vaibhav Hacker/Desktop/MTP/Dataset/testing/features_rgb/'
# FEATURE_DIR_FLOW = 'D:/Vaibhav Hacker/Desktop/MTP/Dataset/testing/features_flow/'

# video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(('.mp4', '.avi', '.mov'))]

# print(f"{'Video Name':30} | {'# Frames':>8} | {'RGB Shape':>20} | {'Flow Shape':>20}")
# print("-" * 90)

# for vid in video_files:
#     # --- Count video frames ---
#     vid_path = os.path.join(VIDEO_DIR, vid)
#     cap = cv2.VideoCapture(vid_path)
#     num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     cap.release()

#     # --- Feature file names ---
#     npy_name = vid.replace('.mp4', '.npy').replace('.avi', '.npy').replace('.mov', '.npy')

#     # --- Load RGB features ---
#     rgb_path = os.path.join(FEATURE_DIR_RGB, npy_name)
#     if os.path.exists(rgb_path):
#         rgb_features = np.load(rgb_path)
#         rgb_shape = rgb_features.shape
#     else:
#         rgb_shape = "Not Found"

#     # --- Load Flow features ---
#     flow_path = os.path.join(FEATURE_DIR_FLOW, npy_name)
#     if os.path.exists(flow_path):
#         flow_features = np.load(flow_path)
#         flow_shape = flow_features.shape
#     else:
#         flow_shape = "Not Found"

#     # --- Print comparison ---
#     print(f"{vid:30} | {num_frames:>8} | {str(rgb_shape):>20} | {str(flow_shape):>20}")

#     # --- Check for mismatches ---
#     if isinstance(rgb_shape, tuple) and isinstance(flow_shape, tuple):
#         if rgb_shape[0] != flow_shape[0]:
#             print(f"⚠️  Mismatch in feature count for: {vid}")


# import cv2

# def get_video_info(video_path):
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print("Error opening video file.")
#         return
    
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     duration = total_frames / fps

#     print(f"📹 Video: {video_path}")
#     print(f"🧮 Total Frames: {total_frames}")
#     print(f"⏱ FPS (frames/sec): {fps}")
#     print(f"🕒 Duration: {duration:.2f} seconds")

#     cap.release()

# get_video_info('D:/Vaibhav Hacker/Desktop/MTP/videos/pick_place/1.avi')



import os

# Path where your .npy files are stored
folder_path = 'D:/Vaibhav Hacker/Desktop/MTP/Dataset/pick_give/features_combined'  # Change this if needed, e.g., './data/'

# Expected file names: 1.npy to 520.npy
expected_files = {f"{i}.npy" for i in range(1, 521)}

# Actual files in the directory
actual_files = set(f for f in os.listdir(folder_path) if f.endswith('.npy'))

# Find missing files
missing_files = sorted(expected_files - actual_files)

# Print missing files
if missing_files:
    print("Missing files:")
    for file in missing_files:
        print(file)
else:
    print("No files are missing.")
