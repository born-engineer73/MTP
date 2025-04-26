import os
from collections import Counter

# Paths
LABEL_DIR = "truth/pick_give"   # Folder with original ground truth files
TRUTH_DIR = "Dataset/pick_give/ground_truth/"   # Folder to save downsampled label files
os.makedirs(TRUTH_DIR, exist_ok=True)

# Process each label file
for file_name in os.listdir(LABEL_DIR):
    if not file_name.endswith(".txt"):
        continue

    input_path = os.path.join(LABEL_DIR, file_name)
    output_path = os.path.join(TRUTH_DIR, file_name)

    with open(input_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]  # remove blank lines

    new_labels = []

    # Group lines into chunks of 8
    for i in range(0, len(lines), 8):
        chunk = lines[i:i+8]
        # print(len(chunk))
        if(len(chunk)!=8):
            continue 
        counts = Counter(chunk)
        # Find label(s) with max count
        max_count = max(counts.values())
        candidates = [label for label, count in counts.items() if count == max_count]

        # Tie-breaker: choose the label that appears first in the chunk
        for label in chunk:
            if label in candidates:
                new_labels.append(label)
                break

    # Save the new labels
    with open(output_path, 'w') as f:
        # for label in new_labels:
        f.write("\n".join(new_labels)+"\n")
    
    print(f"[✓] Processed: {file_name} → {output_path}")
