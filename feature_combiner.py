import numpy as np
import os

def combine_rgb_flow(rgb_path, flow_path, out_path):
    rgb_feat = np.load(rgb_path)   # shape: (T, 1024)
    flow_feat = np.load(flow_path) # shape: (T, 1024)

    if rgb_feat.shape != flow_feat.shape:
        last_flow = flow_feat[-1:]
        flow_feat = np.concatenate([flow_feat, last_flow], axis=0)

        # raise ValueError(f"Shape mismatch: RGB {rgb_feat.shape}, Flow {flow_feat.shape}")

    combined = np.concatenate((rgb_feat, flow_feat), axis=1)  # shape: (T, 2048)
    combined = combined.T
    np.save(out_path, combined)
    print(f"Combined feature saved to: {out_path}")

RGB_DIR = 'D:/Vaibhav Hacker/Desktop/MTP/Dataset/testing2/features_rgb/'
FLOW_DIR = 'D:/Vaibhav Hacker/Desktop/MTP/Dataset/testing2/features_flow/'
COMBINED_DIR = 'D:/Vaibhav Hacker/Desktop/MTP/Dataset/testing2/features_combined/'

os.makedirs(COMBINED_DIR, exist_ok=True)

for fname in os.listdir(RGB_DIR):
    if fname.endswith('.npy'):
        rgb_path = os.path.join(RGB_DIR, fname)
        flow_path = os.path.join(FLOW_DIR, fname)
        out_path = os.path.join(COMBINED_DIR, fname)

        if os.path.exists(flow_path):
            combine_rgb_flow(rgb_path, flow_path, out_path)
        else:
            print(f"Flow file missing for {fname}")
