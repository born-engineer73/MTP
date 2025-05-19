import os

# ==== CONFIGURATION ====
FOLDER_PATH = r'D:\Vaibhav Hacker\Desktop\MTP\Final_MSTCN_repo\data\Training_dataset_3_task\temp'   # Can be video or label folder
# FOLDER_PATH = 'D:/Vaibhav Hacker/Desktop/MTP/Final_MSTCN_repo/data/Training_dataset/testing_ground'   # Can be video or label folder
FILE_TYPE = '.txt'   # Use '.txt' for labels, '.mp4' / '.avi' / '.mov' for videos
START_NUM = 2401
END_NUM = 2520
OFFSET = 1440-START_NUM+1 # e.g., 1 -> 101, 2 -> 102
# ========================

# Get all matching files
existing_files = os.listdir(FOLDER_PATH)
for i in range(START_NUM, END_NUM + 1):
    old_name = f"{i}{FILE_TYPE}"
    new_name = f"{i + OFFSET}{FILE_TYPE}"

    old_path = os.path.join(FOLDER_PATH, old_name)
    new_path = os.path.join(FOLDER_PATH, new_name)

    if os.path.exists(old_path):
        os.rename(old_path, new_path)
        print(f"✅ Renamed: {old_name} → {new_name}")
    else:
        print(f"⚠️ File not found: {old_name}")
